import heapq
import numpy as np
from preconditions import preconditions
import threading
import time

import threadsafe_queue

from runtime.rpc import scheduler_server, scheduler_client

SCHEDULER_PORT = 50051
SLEEP_SECONDS = 2

class Scheduler:
    def __init__(self, policy, get_num_epochs_to_run, min_workers=None):
        # List of worker IDs.
        self._worker_ids = []
        # List of devices.
        self._devices = {}
        # Policy instance.
        self._policy = policy
        # RPC clients.
        self._worker_connections = {}
        # get_num_epochs_to_run function pointer.
        self._get_num_epochs_to_run = get_num_epochs_to_run
        # Minimum number of workers to wait for before scheduling any jobs.
        self._min_workers = 1 if min_workers is None else min_workers
        # Next worker_id to assign.
        self._worker_id_counter = 0
        # Lock to ensure worker_id assignment is thread-safe.
        self._scheduler_lock = threading.Lock()
        # List of available worker IDs.
        self._available_worker_ids = threadsafe_queue.Queue()
        # Throughputs for all current incomplete applications.
        self._throughputs = {}
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Epochs run on each worker_id, for all current incomplete applications.
        self._run_so_far = {}
        # Commands to run for all current incomplete applications.
        self._commands = {}
        # priority_queue for each worker_id.
        self._index = {}

        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self._register_worker,
            'SendHeartbeat': self._handle_heartbeat,
            'Done': self._job_complete,
            }
        self.server_thread = threading.Thread(
            target=scheduler_server.serve,
            args=(port, callbacks,))
        self.server_thread.daemon = True
        self.server_thread.start()

        self._job_id_counter = 0

    def num_workers(self):
        with self._scheduler_lock:
            num_workers = len(self._worker_connections)
        return num_workers

    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_allocation(self):
        # Computes the cluster allocation.
        # self._scheduler_lock must be held when calling this function.
        def flatten(d):
            job_ids = list(d.keys())
            if len(job_ids) == 0:
                return None, None
            worker_ids = list(d[job_ids[0]].keys())
            if len(worker_ids) == 0:
                return None, None
            m = []
            for job_id in job_ids:
                m_row = []
                for worker_id in worker_ids:
                    m_row.append(d[job_id][worker_id])
                m.append(m_row)
            return np.array(m), (job_ids, worker_ids)

        def unflatten(m, index):
            (job_ids, worker_ids) = index
            d = {}
            for i in range(len(job_ids)):
                d[job_ids[i]] = {}
                for j in range(len(worker_ids)):
                    d[job_ids[i]][worker_ids[j]] = m[i][j]
            return d

        flattened_throughputs, index = flatten(self._throughputs)
        if flattened_throughputs is None:
            return None
        flattened_allocation = self._policy.get_allocation(
            flattened_throughputs)
        return unflatten(flattened_allocation, index)

    def _compute_throughput(self, command, worker_id):
        # TODO: compute throughput
        # TODO: add parameter for device_id?
        return 10

    def add_new_job(self, command):
        # Application is a collection of throughputs for each
        # worker_id. (right now, not considering app packing)

        # Public-facing API call to add a new job, updates the
        # internal allocation of workers to jobs.
        # An allocation is of the form {application: <fraction
        # of allocations on different workers.>}. Some scheduler
        # mechanism needs to ensure that each application receives
        # this fraction correctly.
        with self._scheduler_lock:
            job_id = self._job_id_counter
            self._job_id_counter += 1
            self._commands[job_id] = command
            self._run_so_far[job_id] = {}
            self._throughputs[job_id] = {}
            for worker_id in self._worker_ids:
                self._run_so_far[job_id][worker_id] = 0
                self._throughputs[job_id][worker_id] = \
                        self._compute_throughput(command, worker_id)
                heapq.heappush(self._index[worker_id],
                [0.0, 0, job_id])
            self._allocation = self._get_allocation()
        return job_id

    def remove_old_job(self, job_id):
        # Public-facing API call to remove a completed job, updates
        # the internal allocation of workers to jobs.
        with self._scheduler_lock:
            del self._commands[job_id]
            del self._throughputs[job_id]
            del self._run_so_far[job_id]
            if len(self._throughputs) > 0:
                self._allocation = self._get_allocation()
            self._remove_from_index_and_update(job_id)

    @preconditions(lambda self: self._scheduler_lock.locked())
    def _remove_from_index_and_update(self, old_job_id):
        # Computes the cluster allocation.
        # self._scheduler_lock must be held when calling this function.
        for worker_id in self._worker_ids:
            for i in range(len(self._index[worker_id])):
                if self._index[worker_id][i][2] == old_job_id:
                    break
            if len(self._index[worker_id]) > 0:
                self._index[worker_id].pop(i)
                heapq.heapify(self._index[worker_id])

    @preconditions(lambda self: self._scheduler_lock.locked())
    def _add_to_index(self, new_job_id):
        # Computes the cluster allocation.
        # self._scheduler_lock must be held when calling this function.
        for worker_id in self._worker_ids:
            self._index[worker_id].append([0.0, 0, new_job_id])

    @preconditions(lambda self: self._scheduler_lock.locked())
    def _update_index(self):
        # Re-sort keys given that all fractions have decreased but one.
        # TODO: Can optimize this.
        # self._scheduler_lock must be held when calling this function.

        # Stores the fraction of epochs run so far for each job on each worker
        fractions = {}

        # Stores the total number of epochs run for each job
        tot_epochs_run = {}

        for job_id in self._run_so_far:
            fractions[job_id] = {}
            tot_epochs_run[job_id] = 0

        for worker_id in self._worker_ids:
            for job_id in self._run_so_far:
                tot_epochs_run[job_id] += \
                    self._run_so_far[job_id][worker_id]

        for worker_id in self._worker_ids:
            for job_id in self._run_so_far:
                if tot_epochs_run[job_id] == 0:
                    fractions[job_id][worker_id] = 0.0
                else:
                    fractions[job_id][worker_id] = \
                        self._run_so_far[job_id][worker_id] / tot_epochs_run[job_id]
            for i in range(len(self._index[worker_id])):
                [_, _, job_id] = self._index[worker_id][i]
                self._index[worker_id][i][0] = fractions[job_id][worker_id] / \
                    self._allocation[job_id][worker_id]
                self._index[worker_id][i][1] = self._run_so_far[job_id][worker_id]
            heapq.heapify(self._index[worker_id])

    def _register_worker(self, ip_addr, port, devices):
        with self._scheduler_lock:
            worker_id = self._worker_id_counter
            self._worker_ids.append(worker_id)
            self._worker_id_counter += 1
            self._devices[worker_id] = devices
            self._index[worker_id] = []
            self._add_available_worker_id(worker_id)
            self._worker_connections[worker_id] = \
                    scheduler_client.SchedulerRpcClient(ip_addr, port)
            for job_id in self._run_so_far:
                self._run_so_far[job_id][worker_id] = 0
                self._throughputs[job_id][worker_id] = \
                        self._compute_throughput(self._commands[job_id],
                                                 worker_id)
                # TODO: Move this outside the loop?
                # Entries in the index are sorted by
                # fraction_run/fraction_allocated, then number of
                # epochs run, then job_id.
                heapq.heappush(self._index[worker_id], [0.0, 0, job_id])
            self._allocation = self._get_allocation()
            self._update_index()
        return worker_id

    def _handle_heartbeat(self):
        #TODO
        pass

    def _get_available_worker_id(self):
        return self._available_worker_ids.remove()

    def _add_available_worker_id(self, worker_id):
        self._available_worker_ids.add(worker_id)

    def _schedule(self):
        # Schedules the _inactive_ application most in need of an available
        # worker_id (that is, the worker with the lowest
        # fraction_run/fraction_allocated ratio).

        # Scheduler holds two internal data structures,
        # {(application, worker_id): num_epochs_run_on_worker}
        # & {(application, worker_id): allocation_fraction}.
        # As an algorithmic optimization, might be good to maintain
        # a heap of all currently inactive applications for each
        # worker, sorted by fraction_run/fraction_allocated ratio.

        assert self._min_workers is not None and self._min_workers >= 1
        while self.num_workers() < self._min_workers:
            time.sleep(SLEEP_SECONDS)

        worker_id = self._get_available_worker_id()

        with self._scheduler_lock:
            # Get the job_id for this worker_id with minimum
            # fraction_run/fraction_allocated.
            if len(self._index[worker_id]) == 0:
                return None, None, None

            [_, _, job_id] = self._index[worker_id][0]

            self._remove_from_index_and_update(job_id)

            # Number of epochs to run the application on needs to be
            # determined.
            num_epochs = self._get_num_epochs_to_run(job_id, worker_id)

            # Dispatch the job to a worker.
            self._worker_connections[worker_id].run(job_id,
                                                    self._commands[job_id],
                                                    num_epochs)

            return job_id, worker_id, num_epochs

    def _job_complete(self, job_id, worker_id, num_epochs=1):
        # Now, we can update the data structures to reflect the
        # fact that active_application run on a particular worker_id
        # for a certain num_epochs.
        self._add_available_worker_id(worker_id)
        with self._scheduler_lock:
            self._run_so_far[job_id][worker_id] += num_epochs
            self._add_to_index(job_id)
            self._update_index()
