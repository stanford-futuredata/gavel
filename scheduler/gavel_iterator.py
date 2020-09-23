import atexit
import datetime
from filelock import FileLock
import json
from collections.abc import Iterable
import logging
import os
import time
import torch
import traceback
from torch.utils.data.dataloader import DataLoader

from lease import Lease

from runtime.rpc import iterator_client

logging.getLogger('filelock').setLevel(logging.CRITICAL)

INFINITY = (1e9)
LEASE_UPDATE_FRACTION = 0.75
LOG_FORMAT = '[{asctime}] [{event}] [{status}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class GavelIterator:
    def __init__(self, data_loader, checkpoint_dir, load_checkpoint_func,
                 save_checkpoint_func, synthetic_data=False,
                 write_on_close=True, verbose=True):
        if not isinstance(data_loader, Iterable):
            raise ValueError('Data is of uniterable '
                             'type %s' % (type(data_loader)))
        else:
            self._data_loader = data_loader

        self._write_on_close = write_on_close
        atexit.register(self._close_file_handler)
        if self._write_on_close:
            atexit.register(self._write_info)
        self._verbose = verbose
        self._load_checkpoint_func = load_checkpoint_func
        self._save_checkpoint_func = save_checkpoint_func
        self._job_id = int(os.environ['GAVEL_JOB_ID'])
        self._worker_id = int(os.environ['GAVEL_WORKER_ID'])
        self._round_id = int(os.environ['GAVEL_ROUND_ID'])
        self._sched_addr = os.environ['GAVEL_SCHED_ADDR']
        self._sched_port = int(os.environ['GAVEL_SCHED_PORT'])
        self._lock_file = os.path.join(checkpoint_dir, '.gavel.lock')
        self._lock = FileLock(self._lock_file)
        self._gavel_dir = os.path.join(checkpoint_dir, '.gavel')
        self._round_dir = \
            os.path.join(self._gavel_dir, 'round={0}'.format(self._round_id))
        self._worker_dir = \
            os.path.join(self._round_dir, 'worker={0}'.format(self._worker_id))
        with self._lock:
            if not os.path.isdir(self._gavel_dir):
                os.mkdir(self._gavel_dir)
            if not os.path.isdir(self._round_dir):
                os.mkdir(self._round_dir)
        self._log_file = os.path.join(self._round_dir,
                                      'worker={0}.log'.format(self._worker_id))
        self._logger = logging.getLogger('gavel_iterator')
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        self._file_handler = logging.FileHandler(self._log_file)
        self._file_handler.setFormatter(
                logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style='{'))
        self._file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._file_handler)
        self._rpc_client = \
            iterator_client.IteratorRpcClient(self._job_id, self._worker_id,
                                              self._sched_addr,
                                              self._sched_port,
                                              self._logger)
        self._steps = 0
        self._duration = 0
        self._synthetic_data = synthetic_data
        self._done = False
        if self._synthetic_data:
            self._initial_val = None
        self._lease = Lease(0, 0)
        self._update_lease(init=True)
        self._write_info()
        self._prev_time = time.time()

    def __iter__(self):
        self._iterator = iter(self._data_loader)
        return self

    def __next__(self):
        # Update the elapsed time.
        cur_time = time.time()
        elapsed_time = cur_time - self._prev_time
        self._duration += elapsed_time
        self._prev_time = cur_time

        # Update the lease if necessary.
        if (self._steps_until_next_lease_update <= 0 or
            self._time_until_next_lease_update <= 0):
            self._update_lease()

        # Check if the lease has expired.
        lease_expired = (self._duration >= self._lease.max_duration or
                         self._steps >= self._lease.max_steps)
        if lease_expired:
            self._done = True
            self._logger.info(
                '{0} / {1} steps, {2:.4f} / {3:.4f} seconds'.format(
                    self._steps, self._lease.max_steps,
                    self._duration, self._lease.max_duration),
                extra={'event': 'LEASE', 'status': 'EXPIRED'})
            raise StopIteration

        # Return a new data item if one exists.
        try:
            if self._synthetic_data and self._initial_val is not None:
                val = self._initial_val
            else:
                val = next(self._iterator)
                if self._synthetic_data and self._initial_val is None:
                    self._initial_val = val
            self._steps += 1
        except StopIteration as e:
            raise StopIteration

        if self._synthetic_data and self._steps % len(self._data_loader) == 0:
            raise StopIteration

        self._steps_until_next_lease_update -= 1
        self._time_until_next_lease_update -= elapsed_time

        return val

    def __len__(self):
        return len(self._data_loader)

    @property
    def done(self):
        return self._done

    def complete(self):
        self._done = True
        if not self._write_on_close:
            self._write_info()
        self._logger.info('', extra={'event': 'LEASE', 'status': 'COMPLETE'})

    def load_checkpoint(self, *args, **kwargs):
        self._logger.info('', extra={'event': 'LOAD CHECKPOINT',
                                     'status': 'BEGIN'})
        checkpoint = self._load_checkpoint_func(*args, **kwargs)
        self._logger.info('', extra={'event': 'LOAD CHECKPOINT',
                                     'status': 'END'})
        return checkpoint

    def save_checkpoint(self, *args, **kwargs):
        self._logger.info('', extra={'event': 'SAVE CHECKPOINT',
                                     'status': 'BEGIN'})
        retval = self._save_checkpoint_func(*args, **kwargs)
        self._logger.info('', extra={'event': 'SAVE CHECKPOINT',
                                     'status': 'END'})
        return retval

    def _write_info(self):
        self._logger.info('{0}'.format(self._steps),
                          extra={'event': 'PROGRESS', 'status': 'STEPS'})
        self._logger.info('{0}'.format(self._duration),
                          extra={'event': 'PROGRESS', 'status': 'DURATION'})

    def _close_file_handler(self):
        self._logger.removeHandler(self._file_handler)
        self._file_handler.close()

    def _update_lease(self, init=False):
        if init:
            (updated_max_steps, updated_max_duration, extra_time) = \
                self._rpc_client.init()
        else:
            (updated_max_steps, updated_max_duration) = \
                self._rpc_client.update_lease(self._steps,
                                              self._duration,
                                              self._lease.max_steps,
                                              self._lease.max_duration)
            extra_time = 0
        # Update when the next lease update will be. If the lease max steps or
        # max duration has not changed, then assume this will be the final
        # max steps or max duration.
        if updated_max_steps == self._lease.max_steps:
            self._steps_until_next_lease_update = INFINITY
        else:
            next_lease_update_steps = \
                int(updated_max_steps * LEASE_UPDATE_FRACTION +
                    self._lease.max_steps * (1.0 - LEASE_UPDATE_FRACTION))
            self._steps_until_next_lease_update = \
                next_lease_update_steps - self._steps
        if updated_max_duration == self._lease.max_duration:
            self._time_until_next_lease_update = INFINITY
        else:
            next_lease_update_time = \
                (updated_max_duration * LEASE_UPDATE_FRACTION +
                 self._lease.max_duration * (1.0 - LEASE_UPDATE_FRACTION))
            self._time_until_next_lease_update = \
                next_lease_update_time - self._duration + extra_time

        # Update the lease.
        self._lease.max_steps = updated_max_steps
        self._lease.max_duration = updated_max_duration + extra_time
