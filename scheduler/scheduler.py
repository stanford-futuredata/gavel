from __future__ import print_function

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
        # Next job_id to assign.
        self._job_id_counter = 0
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
        # The total number of epochs to run each job for.
        self._total_epochs = {}
        # Epochs run on each worker_id, for all current incomplete applications.
        self._epochs_run_so_far = {}
        # Time run so far on each worker_id, for all current incomplete applications.
        self._time_run_so_far = {}
        # Commands to run for all current incomplete applications.
        self._commands = {}
        # priority_queue for each worker_id.
        self._index = {}

        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'SendHeartbeat': self._send_heartbeat_callback,
            'Done': self._done_callback,
        }
        self.server_thread = threading.Thread(
            target=scheduler_server.serve,
            args=(port, callbacks))
        self.server_thread.daemon = True
        self.server_thread.start()

        self.scheduler_thread = threading.Thread(
            target=self._schedule,
            args=())
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()


    def add_job(self, command, total_epochs):
        """Adds a new job to the scheduler.

        Enables users to schedule a new job. Updates the internal
        allocation of workers to jobs. An allocation is of the form
        {job: <fraction of allocations on different workers.>}.

        Args:
            command: The command to execute.
            total_epochs: The total number of epochs to run the command for.

        Returns:
            The job_id of the newly added job.
        """

        with self._scheduler_lock:
            job_id = self._job_id_counter
            self._job_id_counter += 1
            self._commands[job_id] = command
            self._epochs_run_so_far[job_id] = {}
            self._time_run_so_far[job_id] = {}
            self._total_epochs[job_id] = total_epochs
            self._throughputs[job_id] = {}
            for worker_id in self._worker_ids:
                self._epochs_run_so_far[job_id][worker_id] = 0
                self._time_run_so_far[job_id][worker_id] = 0.0
                self._throughputs[job_id][worker_id] = \
                    self._compute_throughput(command, worker_id)
                heapq.heappush(self._index[worker_id],
                               [0.0, 0, job_id])
            self._allocation = self._get_allocation()
        return job_id


    def remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The id of the job to remove.
        """

        with self._scheduler_lock:
            self._remove_job(job_id)


    def num_workers(self):
        """Returns the number of workers the scheduler is connected to."""

        with self._scheduler_lock:
            return len(self._worker_connections)


    def num_jobs(self):
        """Returns the number of jobs the scheduler is currently managing."""

        with self._scheduler_lock:
            return len(self._epochs_run_so_far)


    def _schedule(self):
        """Schedules jobs on workers.

        In a loop, schedules the inactive application most in need of an
        available worker (that is, the worker with the lowest
        fraction_run/fraction_allocated ratio).

        Scheduler holds two internal data structures,
        {(application, worker_id): num_epochs_run_on_worker}
        & {(application, worker_id): allocation_fraction}.
        As an algorithmic optimization, might be good to maintain
        a heap of all currently inactive applications for each
        worker, sorted by fraction_run/fraction_allocated ratio.
        """

        # TODO: change to exception
        assert self._min_workers is not None and self._min_workers >= 1
        while self.num_workers() < self._min_workers:
            time.sleep(SLEEP_SECONDS)

        while True:
            worker_id = self._get_available_worker_id()
            with self._scheduler_lock:
                if len(self._index[worker_id]) == 0:
                    # NOTE: do we need to add the worker_id back here?
                    continue
                [_, _, job_id] = self._index[worker_id][0]
                self._remove_from_index_and_update(job_id)
                num_epochs = self._get_num_epochs_to_run(job_id, worker_id)
                self._worker_connections[worker_id].run(job_id,
                                                        self._commands[job_id],
                                                        num_epochs)


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_allocation(self):
        """Computes the allocation.

        Uses the specified policy to compute an allocation of jobs to
        compute resources. Requires self._scheduler_lock to be held
        when calling this function.

        Returns:
            A 2-level dict indexed by job_id and then worker_id. For
            example,

            {0: {0: 0.25, 1: 0.95}, 1: {0: 0.75, 1: 0.05}}

            indicates that for 25% of the time, worker 0 should run job 0,
            and for 95% of the time, worker 1 should run job 0.
        """

        def flatten(d):
            """Converts a 2-level dict to a NumPy array."""

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
            """Converts a NumPy array to a 2-level dict."""

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


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _remove_job(self, job_id):
        """Removes internal state associated with a job.

        Internal implementation for removing a job.
        Requires self._scheduler_lock to be held when calling this function.

       Args:
           job_id: The id of the job to remove.
       """

        del self._commands[job_id]
        del self._throughputs[job_id]
        del self._epochs_run_so_far[job_id]
        del self._time_run_so_far[job_id]
        del self._total_epochs[job_id]
        if len(self._throughputs) > 0:
            self._allocation = self._get_allocation()
        self._remove_from_index_and_update(job_id)


    def _get_available_worker_id(self):
        """Returns the worker_id of the next available worker."""

        return self._available_worker_ids.remove()


    def _add_available_worker_id(self, worker_id):
        """Adds a worker_id to the list of available workers."""

        self._available_worker_ids.add(worker_id)


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _add_to_index(self, job_id):
        """Adds a job_id to each worker.

       Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' indexes.
        """

        for worker_id in self._worker_ids:
            self._index[worker_id].append([0.0, 0, job_id])


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _remove_from_index_and_update(self, job_id):
        # Computes the cluster allocation.
        # self._scheduler_lock must be held when calling this function.
        for worker_id in self._worker_ids:
            for i in range(len(self._index[worker_id])):
                if self._index[worker_id][i][2] == job_id:
                    if len(self._index[worker_id]) > 0:
                        self._index[worker_id].pop(i)
                        heapq.heapify(self._index[worker_id])
                    break


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _update_index(self):
        """Updates the index of each worker.

        Re-sorts the index of each worker to compute the next job to run.
        For a given worker w_i, the next job to be scheduled will be the job
        that has so far received the smallest fraction of its computed
        fair allocation.
        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' indexes.
        """

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        # Stores the total amount of time run on each worker among currently
        # running jobs.
        tot_time_run = {}

        for worker_id in self._worker_ids:
            fractions[worker_id] = {}
            tot_time_run[worker_id] = self._get_total_time_run(worker_id)

        for worker_id in self._worker_ids:
            for job_id in self._time_run_so_far:
                if tot_time_run[worker_id] == 0.0:
                    fractions[worker_id][job_id] = 0.0
                else:
                    fraction = self._time_run_so_far[job_id][worker_id] / \
                        tot_time_run[worker_id]
                    fractions[worker_id][job_id] = fraction
            for i in range(len(self._index[worker_id])):
                [_, _, job_id] = self._index[worker_id][i]
                self._index[worker_id][i][0] = fractions[worker_id][job_id] / \
                    self._allocation[job_id][worker_id]
                self._index[worker_id][i][1] = \
                    self._epochs_run_so_far[job_id][worker_id]
            heapq.heapify(self._index[worker_id])


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_total_epochs_run(self, job_id):
        # TODO: change to exception
        assert(job_id in self._epochs_run_so_far)
        total_epochs_run = 0
        for worker_id in self._epochs_run_so_far[job_id]:
            total_epochs_run += self._epochs_run_so_far[job_id][worker_id]
        return total_epochs_run


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_total_time_run(self, worker_id):
        total_time_run = 0.0
        for job_id in self._time_run_so_far:
            total_time_run += self._time_run_so_far[job_id][worker_id]
        return total_time_run


    def _register_worker_callback(self, ip_addr, port, devices):
        """Registers a worker with the scheduler.

        Initializes state for a new worker and assigns it an id.
        The worker provides an IP address and port for its RPC server
        so that the scheduler can establish an RPC client for
        scheduler-to-worker communication. The worker also
        enumerates its available devices so that the scheduler
        can make fine-grained scheduling decisions.

        Args:
            ip_addr: IP address of the worker's RPC server.
            port: Port number for the worker's RPC server.
            devices: List of available devices on the worker.

        Returns:
            The worker_id of the newly registered worker.
        """

        with self._scheduler_lock:
            worker_id = self._worker_id_counter
            self._worker_ids.append(worker_id)
            self._worker_id_counter += 1
            self._devices[worker_id] = devices
            self._index[worker_id] = []
            self._add_available_worker_id(worker_id)
            self._worker_connections[worker_id] = \
                    scheduler_client.SchedulerRpcClient(ip_addr, port)
            for job_id in self._epochs_run_so_far:
                self._epochs_run_so_far[job_id][worker_id] = 0
                self._time_run_so_far[job_id][worker_id] = 0.0
                self._throughputs[job_id][worker_id] = \
                    self._compute_throughput(self._commands[job_id],
                                             worker_id)
                # Entries in the index are sorted by
                # fraction_run/fraction_allocated, then number of
                # epochs run, then job_id.
                heapq.heappush(self._index[worker_id], [0.0, 0, job_id])
            self._allocation = self._get_allocation()
            self._update_index()
        return worker_id


    def _send_heartbeat_callback(self):
        #TODO
        pass


    def _done_callback(self, job_id, worker_id, execution_time, num_epochs=1):
        """Handles completion of a scheduled job.

        Updates the running total of completed epochs and time spent on each
        worker, for every currently active application. Removes the job from
        the scheduler if the job has finished all its requested epochs. Adds
        the worker back to the list of available workers.

        Args:
            job_id: The id of the completed job.
            worker_id: The id of the worker where the job was completed.
            num_epochs: The number of epochs the job ran for.
        """

        with self._scheduler_lock:
            self._epochs_run_so_far[job_id][worker_id] += num_epochs
            self._time_run_so_far[job_id][worker_id] += execution_time
            print("[Completed] Job ID: %d, Worker ID: %d" % (job_id, worker_id))
            print("[{job_id: {worker_id: epochs}}]", self._epochs_run_so_far) # NOTE: for debug purposes
            print("[{job_id: {worker_id: time}}]", self._time_run_so_far) # NOTE: for debug purposes
            print()
            if self._get_total_epochs_run(job_id) < self._total_epochs[job_id]:
                self._add_to_index(job_id)
            else:
                self._remove_job(job_id)
            self._update_index()
        self._add_available_worker_id(worker_id)
