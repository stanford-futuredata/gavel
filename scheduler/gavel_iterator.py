from filelock import FileLock
import json
from collections.abc import Iterable
import os
import time
import torch
import traceback
from torch.utils.data.dataloader import DataLoader

from lease import Lease

from runtime.rpc import iterator_client

INFINITY = (1e9)
LEASE_UPDATE_FRACTION = 0.75

class GavelIterator:
    def __init__(self, data_loader, gavel_dir, load_checkpoint_func,
                 save_checkpoint_func, synthetic_data=False, verbose=True):
        if not isinstance(data_loader, Iterable):
            raise ValueError('Data is of uniterable '
                             'type %s' % (type(data_loader)))
        else:
            self._data_loader = data_loader

        self._verbose = verbose
        self._load_checkpoint_func = load_checkpoint_func
        self._save_checkpoint_func = save_checkpoint_func
        self._job_id = int(os.environ['GAVEL_JOB_ID'])
        self._worker_id = int(os.environ['GAVEL_WORKER_ID'])
        self._sched_addr = os.environ['GAVEL_SCHED_ADDR']
        self._sched_port = int(os.environ['GAVEL_SCHED_PORT'])
        self._lock_file = os.path.join(gavel_dir, '.gavel.lock')
        self._gavel_file = os.path.join(gavel_dir, '.gavel.json')
        self._lock = FileLock(self._lock_file)
        self._rpc_client = \
            iterator_client.IteratorRpcClient(self._job_id, self._worker_id,
                                              self._sched_addr,
                                              self._sched_port)
        self._steps = 0
        self._duration = 0
        self._synthetic_data = synthetic_data
        self._done = False
        if self._synthetic_data:
            self._initial_val = None
        # TODO: Tie this with loading the checkpoint
        self._lease = Lease(0, 0)
        self._update_lease(init=True)
        self._write_info()
        self._prev_time = time.time()

    def __del__(self):
        self.complete()

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
        if self._duration >= self._lease.max_duration:
            if self._verbose:
                print('Gavel lease expired: %f seconds '
                      '(max %f seconds)' % (self._duration,
                                            self._lease.max_duration))
            self.complete()
            raise StopIteration
        elif self._steps >= self._lease.max_steps:
            if self._verbose:
                print('Gavel lease expired: %d steps '
                      '(max %d steps)' % (self._steps,
                                          self._lease.max_steps))
            self.complete()
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
            # TODO: Enforce contract that application calls complete before
            # exiting.
            self._write_info()
            raise StopIteration

        if self._synthetic_data and self._steps % len(self._data_loader) == 0:
            # TODO: Enforce contract that application calls complete before
            # exiting.
            self._write_info()
            raise StopIteration

        self._steps_until_next_lease_update -= 1
        self._time_until_next_lease_update -= elapsed_time

        return val

    def __len__(self):
        return len(self._data_loader)

    @property
    def done(self):
        return self._done

    def load_checkpoint(self, *args, **kwargs):
        self._load_checkpoint_func(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        self._save_checkpoint_func(*args, **kwargs)

    def complete(self):
        self._done = True
        self._write_info()

    def _write_info(self):
        job_id = str(self._job_id)
        worker_id = str(self._worker_id)
        with self._lock:
            try:
                if os.path.exists(self._gavel_file):
                    with open(self._gavel_file, 'r') as f:
                        gavel_info = json.load(f)
                else:
                    gavel_info = {}
                if job_id not in gavel_info:
                    gavel_info[job_id] = {}
                if worker_id not in gavel_info[job_id]:
                    gavel_info[job_id][worker_id] = {}
                gavel_info[job_id][worker_id]['steps'] = self._steps
                gavel_info[job_id][worker_id]['duration'] = self._duration
                print('Gavel info:\n{0}'.format(
                        json.dumps(gavel_info, indent=2)))
                with open(self._gavel_file, 'w') as f:
                    json.dump(gavel_info, f)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError('Could not write Gavel info!')

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
