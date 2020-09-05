from filelock import FileLock
import json
from collections.abc import Iterable
import os
import time
import torch
from torch.utils.data.dataloader import DataLoader

from lease import Lease

from runtime.rpc import iterator_client

INFINITY = (1e9)
LEASE_UPDATE_FRACTION = 0.75

class GavelIterator:
    def __init__(self, data_loader, gavel_dir, synthetic_data=False,
                 verbose=True):
        if not isinstance(data_loader, Iterable):
            raise ValueError('Data is of uniterable '
                             'type %s' % (type(data_loader)))
        else:
            self._data_loader = data_loader

        self._verbose = verbose
        self._gpu_id = torch.cuda.current_device()
        self._lock_file = os.path.join(gavel_dir, '.gavel.lock')
        self._gavel_file = os.path.join(gavel_dir, '.gavel.json')
        self._lock = FileLock(self._lock_file)
        self._read_config()
        self._rpc_client = \
            iterator_client.IteratorRpcClient(self._job_id, self._worker_id,
                                              self._server_addr,
                                              self._server_port)
        self._steps = 0
        self._duration = 0
        self._synthetic_data = synthetic_data
        self._done = False
        if self._synthetic_data:
            self._initial_val = None
        # TODO: Tie this with loading the checkpoint
        self._lease = Lease(0, 0)
        self._update_lease(init=True)
        self._prev_time = time.time()
        self._write_info()

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

    def complete(self):
        self._done = True
        self._write_info()

    def _read_config(self):
        if self._verbose:
            print('Trying to read config info '
                  'for GPU {0}...'.format(self._gpu_id))
        gpu_id = str(self._gpu_id)
        with self._lock:
            try:
                with open(self._gavel_file, 'r') as f:
                    gavel_info = json.load(f)
                    self._job_id = \
                        gavel_info['job_info'][gpu_id]['job_id']
                    self._worker_id = \
                        gavel_info['job_info'][gpu_id]['worker_id']
                    self._server_addr = gavel_info['server_info']['addr']
                    self._server_port = gavel_info['server_info']['port']
            except Exception as e:
                raise RuntimeError('Could not read Gavel info: {0}'.format(e))

    def _write_info(self):
        gpu_id = str(self._gpu_id)
        with self._lock:
            try:
                with open(self._gavel_file, 'r') as f:
                    gavel_info = json.load(f)
                gavel_info['job_info'][gpu_id]['steps'] = self._steps
                gavel_info['job_info'][gpu_id]['duration'] = self._duration
                with open(self._gavel_file, 'w') as f:
                    json.dump(gavel_info, f)
            except Exception as e:
                raise RuntimeError('Could not write Gavel info: {0}'.format(e))

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
