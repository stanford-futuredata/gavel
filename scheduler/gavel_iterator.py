from collections.abc import Iterable
import os
import time
import torch
from torch.utils.data.dataloader import DataLoader

from lease import Lease

from runtime.rpc import iterator_client

LEASE_UPDATE_FRACTION = 0.75

class GavelIterator:
    def __init__(self, data_loader, job_id, worker_id, distributed,
                 server_addr, server_port, gavel_dir, synthetic_data=False,
                 verbose=True):
        if not isinstance(data_loader, Iterable):
            raise ValueError('Data is of uniterable '
                             'type %s' % (type(data_loader)))
        else:
            self._data_loader = data_loader
        self._rpc_client = iterator_client.IteratorRpcClient(job_id, worker_id,
                                                             server_addr,
                                                             server_port)
        self._job_id = job_id
        self._worker_id = worker_id
        self._steps = 0
        self._duration = 0
        self._distributed = distributed
        self._done = False
        self._synthetic_data = synthetic_data
        self._verbose = verbose
        if self._synthetic_data:
            self._initial_val = None
        assert(os.path.isdir(os.path.join(gavel_dir)))
        self._info_file = os.path.join(gavel_dir, '.gavel_info')

        # TODO: Tie this with loading the checkpoint
        initial_max_steps, initial_max_duration = self._init()
        self._lease = Lease(initial_max_steps, initial_max_duration)
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

    def _write_info(self):
        try:
            with open(self._info_file, 'w') as f:
                f.write('%d\n%f' % (self._steps, self._duration))
        except Exception as e:
            if self._verbose:
                print('Error writing info to \"%s\": %s' % (self._info_file,
                                                            e))

    def _init(self):
        self._rpc_client.init()

    def _update_lease(self):
        (updated_max_steps, updated_max_duration) = \
            self._rpc_client.update_lease(self._steps,
                                          self._duration,
                                          self._lease.max_steps,
                                          self._lease.max_duration)
        next_lease_update_steps = \
            int(updated_max_steps * LEASE_UPDATE_FRACTION +
                self._lease.max_steps * (1.0 - LEASE_UPDATE_FRACTION))
        next_lease_update_time = \
            (updated_max_duration * LEASE_UPDATE_FRACTION +
             self._lease.max_duration * (1.0 - LEASE_UPDATE_FRACTION))
        self._steps_until_next_lease_update = \
            next_lease_update_steps - self._steps
        self._time_until_next_lease_update = \
            next_lease_update_time - self._duration
        self._lease.max_steps = updated_max_steps
        self._lease.max_duration = updated_max_duration
