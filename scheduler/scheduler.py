import heapq
import numpy as np

import runtime
import threadsafe_queue

class Scheduler:
    def __init__(self, worker_ids, policy, stub, get_num_epochs_to_run,
                 run_server=False):
        # List of worker IDs.
        self._worker_ids = worker_ids
        # Policy instance.
        self._policy = policy
        # Worker stub instance.
        self._stub = stub
        # get_num_epochs_to_run function pointer.
        self._get_num_epochs_to_run = get_num_epochs_to_run

        # List of available worker IDs.
        self._available_worker_ids = threadsafe_queue.Queue()
        for worker_id in worker_ids:
            self._available_worker_ids.add(worker_id)
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
        for worker_id in worker_ids:
            self._index[worker_id] = []

        if run_server:
            self.server_thread = threading.Thread(
                runtime.rpc.scheduler_server.serve,
                args=(self._available_worker_ids,))
            self.server_thread.daemon = True
            self.server_thread.start()

        self._last_job_id_assigned = 0

    def _get_allocation(self):
        def flatten(d):
            job_ids = list(d.keys())
            worker_ids = list(d[job_ids[0]].keys())
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
        flattened_allocation = self._policy.get_allocation(
            flattened_throughputs)
        return unflatten(flattened_allocation, index)

    def add_new_job(self, throughputs, command):
        # Application is a collection of throughputs for each
        # worker_id. (right now, not considering app packing)

        # Public-facing API call to add a new job, updates the
        # internal allocation of workers to jobs.
        # An allocation is of the form {application: <fraction
        # of allocations on different workers.>}. Some scheduler
        # mechanism needs to ensure that each application receives
        # this fraction correctly.
        job_id = self._last_job_id_assigned
        self._commands[job_id] = command
        self._last_job_id_assigned += 1
        self._throughputs[job_id] = throughputs
        self._allocation = self._get_allocation()
        self._run_so_far[job_id] = {}
        for worker_id in self._worker_ids:
            self._run_so_far[job_id][worker_id] = 0
            # Entries in the index are sorted by fraction_run/fraction_allocated,
            # then number of epochs run, then job_id.
            heapq.heappush(self._index[worker_id],
                           [0.0, 0, job_id])
        return job_id

    def remove_old_job(self, job_id):
        # Public-facing API call to remove a completed job, updates
        # the internal allocation of workers to jobs.
        del self._commands[job_id]
        del self._throughputs[job_id]
        del self._run_so_far[job_id]
        self._allocation = self._get_allocation()
        self._remove_from_index_and_update(job_id)

    def _remove_from_index_and_update(self, old_job_id):
        for worker_id in self._worker_ids:
            for i in range(len(self._index[worker_id])):
                if self._index[worker_id][i][2] == old_job_id:
                    break
            self._index[worker_id].pop(i)
            heapq.heapify(self._index[worker_id])

    def _add_to_index_and_update(self, new_job_id):
        # Re-sort keys given that all fractions have decreased but one.
        # TODO: Can optimize this.
        fractions = {}
        tot_epochs_run = {}
        for job_id in self._run_so_far:
            fractions[job_id] = {}
            tot_epochs_run[job_id] = 0
        for worker_id in self._worker_ids:
            for job_id in self._run_so_far:
                tot_epochs_run[job_id] += \
                    self._run_so_far[job_id][worker_id]
        for resouce_type in self._worker_ids:
            for job_id in self._run_so_far:
                if tot_epochs_run[job_id] == 0:
                    fractions[job_id][worker_id] = 0.0
                else:
                    fractions[job_id][worker_id] = \
                        self._run_so_far[job_id][worker_id] / tot_epochs_run[job_id]
            self._index[worker_id].append([0.0, 0, new_job_id])
            for i in range(len(self._index[worker_id])):
                [_, _, job_id] = self._index[worker_id][i]
                self._index[worker_id][i][0] = fractions[job_id][worker_id] / \
                    self._allocation[job_id][worker_id]
                self._index[worker_id][i][1] = self._run_so_far[job_id][worker_id]
            heapq.heapify(self._index[worker_id])

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

        worker_id = self._get_available_worker_id()

        # Get the job_id for this worker_id with minimum
        # fraction_run/fraction_allocated.
        [_, _, job_id] = self._index[worker_id][0]
        self._remove_from_index_and_update(job_id)

        # Number of epochs to run the application on needs to be
        # determined.
        num_epochs = self._get_num_epochs_to_run(job_id,
                                                 worker_id)
        self._stub.run_application(self._commands[job_id], job_id, worker_id,
                                   num_epochs)
        return job_id, worker_id, num_epochs

    def _schedule_callback(self, job_id, worker_id, num_epochs):
        # Now, we can update the data structures to reflect the
        # fact that active_application run on a particular worker_id
        # for a certain num_epochs.
        self._run_so_far[job_id][worker_id] += num_epochs

        self._add_to_index_and_update(job_id)
