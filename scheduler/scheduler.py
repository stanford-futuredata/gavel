from __future__ import print_function

import heapq
import numpy as np
from preconditions import preconditions
import sys
import threading
import time

import priority_queue
from runtime.rpc import scheduler_server, scheduler_client
import utils

SCHEDULER_PORT = 50051
SLEEP_SECONDS = 2

class Scheduler:

    def __init__(self, policy, get_num_steps_to_run, emulate=False,
                 normalizing_worker_type=None, throughputs_directory=None):
        # Emulate flag.
        self._emulate = emulate

        # Datastructures to faithfully emulate.
        # Latest emulated timestamp.
        self._timestamp = 0
        # Start and last processed timestamp for each job_id.
        self._per_job_start_timestamps = {}
        self._per_job_timestamps = {}
        # Job completion times.
        self._job_completion_times = {}
        # Queue of events that need to be processed at specific timestamps.
        self._event_queue = []

        # List of worker IDs.
        self._worker_ids = []
        # List of worker types.
        self._worker_types = set()
        # Mapping of worker ID to worker type.
        self._worker_id_to_worker_type_mapping = {}
        # List of devices.
        self._devices = {}
        # Policy instance.
        self._policy = policy
        # RPC clients.
        self._num_workers = 0
        self._worker_connections = {}
        # get_num_steps_to_run function pointer.
        self._get_num_steps_to_run = get_num_steps_to_run
        # Next job_id to assign.
        self._job_id_counter = 0
        # Next worker_id to assign.
        self._worker_id_counter = 0
        # Lock to ensure worker_id assignment is thread-safe.
        self._scheduler_lock = threading.Lock()
        # List of available worker IDs.
        self._available_worker_ids = priority_queue.Queue()
        # Throughputs for all current incomplete applications.
        self._throughputs = {}
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Epochs run on each worker_id, for all current incomplete applications.
        self._steps_run_so_far = {}
        # Time run so far on each worker_id, for all current incomplete applications.
        self._time_run_so_far = {}
        # Number of jobs to compute fair share.
        self._num_jobs = 0
        # Commands to run for all current incomplete applications.
        self._jobs = {}
        # Priority queues for each worker_type.
        self._per_worker_type_job_queue = {}
        # Normalizing worker type.
        self.normalizing_worker_type = normalizing_worker_type
        # Throughputs for all job types (pre-measured).
        if throughputs_directory is not None:
            self._all_throughputs = utils.read_all_throughputs(
                throughputs_directory)
        else:
            self._all_throughputs = {}

        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'SendHeartbeat': self._send_heartbeat_callback,
            'Done': self._done_callback,
        }
        if not self._emulate:
            self.server_thread = threading.Thread(
                target=scheduler_server.serve,
                args=(port, callbacks))
            self.server_thread.daemon = True
            self.server_thread.start()

            self.start_scheduling_thread()


    def start_scheduling_thread(self):
        self.scheduler_thread = threading.Thread(
            target=self._schedule,
            args=())
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

    """
    ======================================================================
       Methods for emulation.
    ======================================================================
    """

    @preconditions(lambda self: self._emulate)
    def add_to_event_queue(self, func, args, timestamp):
        """Adds passed-in func to the event queue with the passed-in timestamp.

        Used in emulation mode to queue add_job and register_worker events."""

        with self._scheduler_lock:
            self._event_queue.append((timestamp, func, args))
            self._event_queue.sort(key=lambda x: x[0])

    @preconditions(lambda self: self._emulate)
    def execute_from_event_queue(self, timestamp):
        """Executes all events that have a timestamp lower than the current timestamp."""

        while True:
            with self._scheduler_lock:
                if len(self._event_queue) == 0:
                    return timestamp
                # If passed-in timestamp is before the first timestamp in the event queue,
                # and no jobs are scheduled to run.
                if timestamp < self._event_queue[0][0] and len(self._steps_run_so_far) > 0:
                    return timestamp
                (timestamp, func, args) = self._event_queue.pop(0)
            func(*args)

    """
    ======================================================================
       Public-facing scheduler methods.
    ======================================================================
    """

    def add_job(self, job):
        """Adds a new job to the scheduler.

        Enables users to schedule a new job. Updates the internal
        allocation of workers to jobs. An allocation is of the form
        {job: <fraction of allocations on different workers>}.

        Args:
            command: The command to execute.
            total_steps: The total number of steps to run the command for.

        Returns:
            The job_id of the newly added job.
        """

        with self._scheduler_lock:
            job_id = self._job_id_counter
            self._job_id_counter += 1
            job._job_id = job_id
            self._jobs[job_id] = job
            self._steps_run_so_far[job_id] = {}
            self._time_run_so_far[job_id] = {}
            self._throughputs[job_id] = {}
            for worker_type in self._worker_types:
                self._steps_run_so_far[job_id][worker_type] = 0
                self._throughputs[job_id][worker_type] = \
                    self._compute_throughput(job, worker_type)

            self._reset_time_run_so_far()
            self._add_to_queue(job_id)
            self._allocation = self._get_allocation()
            if self._emulate:
                self._per_job_start_timestamps[job_id] = self._timestamp
            else:
                self._per_job_start_timestamps[job_id] = time.time()
        return job_id


    def remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        with self._scheduler_lock:
            duration = self._per_job_timestamps[job_id] - \
                self._per_job_start_timestamps[job_id]
            self._job_completion_times[job_id] = duration
            print("Job %d completed\n\tStart timestamp: %.2f\n\t"
                  "End timestamp: %.2f\nDuration: %.2f %s\n" % (
                      job_id,
                      self._per_job_start_timestamps[job_id],
                      self._per_job_timestamps[job_id],
                      duration,
                      "timeunits" if self._emulate else "seconds")
                  )
            del self._jobs[job_id]
            del self._steps_run_so_far[job_id]
            del self._time_run_so_far[job_id]
            del self._throughputs[job_id]

            self._reset_time_run_so_far()
            self._remove_from_queue(job_id)
            if len(self._throughputs) > 0:
                self._allocation = self._get_allocation()


    def num_workers(self):
        """Returns the number of workers the scheduler is connected to."""

        with self._scheduler_lock:
            return self._num_workers


    def is_done(self):
        """Returns whether the scheduler is done with all of its assigned work."""
        with self._scheduler_lock:
            return len(self._event_queue) == 0 and len(self._steps_run_so_far) == 0


    def shutdown(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            if self._emulate:
                print("Total time taken: %.2f timeunits" % self._timestamp)
            print("Job completion times:\n\t%s" % self._job_completion_times)
            average_job_completion_time = sum(self._job_completion_times.values()) / \
                len(self._job_completion_times)
            unit = "timeunits" if self._emulate else "seconds"
            print("Average job completion time: %.3f %s" % (average_job_completion_time,
                                                            unit))
            for worker_id in self._worker_connections:
                self._worker_connections[worker_id].shutdown()
        # TODO: Any other cleanup?
        sys.exit()


    """
    ======================================================================
       Scheduler's main _schedule() method.
    ======================================================================
    """

    def _schedule(self):
        """Schedules jobs on workers.

        In a loop, schedules the inactive application most in need of an
        available worker (that is, the worker with the lowest
        fraction_run/fraction_allocated ratio).

        Scheduler holds two internal data structures,
        {(application, worker_type): time_run_on_worker}
        & {(application, worker_type): allocation_fraction}.
        As an algorithmic optimization, the scheduler maintains
        a heap of all currently inactive applications for each
        worker, sorted by fraction_run/fraction_allocated ratio.
        """

        while True:
            timestamp, worker_id = self._remove_available_worker_id()
            if self._emulate:
                timestamp = self.execute_from_event_queue(timestamp)
            with self._scheduler_lock:
                worker_type = self._worker_id_to_worker_type_mapping[worker_id]
                self._update_queue()
                if len(self._per_worker_type_job_queue[worker_type]) == 0:
                    if not self._emulate:
                        timestamp = time.time()
                    self._add_available_worker_id(worker_id, timestamp)
                    continue
                [priority, _, job_id] = self._per_worker_type_job_queue[worker_type][0]

                # Get available worker_id with the highest priority for this particular job_id.
                highest_priority, worker_id_with_highest_priority = self._get_highest_priority(job_id)
                if priority > highest_priority:  # Lower is better.
                    timestamp_with_highest_priority, worker_id_with_highest_priority = \
                        self._remove_available_worker_id(worker_id=worker_id_with_highest_priority)
                    # If removal succeeded.
                    if worker_id_with_highest_priority is not None:
                        self._add_available_worker_id(worker_id, timestamp)
                        worker_id = worker_id_with_highest_priority
                        timestamp = timestamp_with_highest_priority
                        worker_type = self._worker_id_to_worker_type_mapping[worker_id]

                timestamp = max(timestamp, self._per_job_timestamps.get(job_id, 0))
                self._remove_from_queue(job_id)
                num_steps = self._get_num_steps_to_run(job_id, worker_type)
                if not self._emulate:
                    self._worker_connections[worker_id].run([(job_id,
                                                              self._jobs[job_id].command(),
                                                              num_steps)])
            # Can only call _done_callback with lock released.
            if self._emulate:
                # When emulating, directly call _done_callback since there's no worker.
                duration = self._jobs[job_id].duration()
                if self.normalizing_worker_type is not None:
                    normalizing_factor = self._throughputs[job_id][worker_type] / \
                        self._throughputs[job_id][self.normalizing_worker_type]
                    duration /= normalizing_factor
                # TODO: change to exception.
                assert duration is not None
                self._done_callback(job_id, worker_id,
                                    duration,
                                    timestamp=timestamp+duration)
                self._timestamp = max(self._timestamp, timestamp+duration)
                self._per_job_timestamps[job_id] = timestamp + duration

    """
    ======================================================================
       Helper methods to compute each user's fair allocation.
    ======================================================================
    """

    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_allocation(self):
        """Computes the allocation.

        Uses the specified policy to compute an allocation of jobs to
        compute resources. Requires self._scheduler_lock to be held
        when calling this function.

        Returns:
            A 2-level dict indexed by job_id and then worker_type. For
            example,

            {0: {"v100": 0.25, "p100": 0.95}, 1: {"v100": 0.75, "p100": 0.05}}

            indicates that for 25% of the time, worker type 'v100' should run job 0,
            and for 95% of the time, worker type 'p100' should run job 0.
        """

        def flatten(d):
            """Converts a 2-level dict to a NumPy array."""

            job_ids = list(d.keys())
            if len(job_ids) == 0:
                return None, None
            worker_types = list(d[job_ids[0]].keys())
            if len(worker_types) == 0:
                return None, None
            m = []
            for job_id in job_ids:
                m_row = []
                for worker_type in worker_types:
                    m_row.append(d[job_id][worker_type])
                m.append(m_row)
            return np.array(m), (job_ids, worker_types)

        def unflatten(m, index):
            """Converts a NumPy array to a 2-level dict."""

            (job_ids, worker_types) = index
            d = {}
            for i in range(len(job_ids)):
                d[job_ids[i]] = {}
                for j in range(len(worker_types)):
                    d[job_ids[i]][worker_types[j]] = m[i][j]
            return d

        flattened_throughputs, index = flatten(self._throughputs)
        if flattened_throughputs is None:
            return None
        flattened_allocation = self._policy.get_allocation(
            flattened_throughputs)
        unflattened_allocation = unflatten(flattened_allocation, index)
        print("New allocation\n\t%s\n" % unflattened_allocation)
        return unflattened_allocation


    def _compute_throughput(self, job, worker_type):
        job_type = job.job_type()
        job_type = tuple([job_type])
        if job_type in self._all_throughputs and worker_type in self._all_throughputs[job_type]:
            throughput = self._all_throughputs[job_type][worker_type]
            return throughput[0]
        # TODO: compute throughput.
        # TODO: add parameter for device_id?
        return 10

    """
    ======================================================================
       Methods to update the scheduler's internal data structures.
    ======================================================================
    """

    @preconditions(lambda self: self._scheduler_lock.locked())
    def _reset_time_run_so_far(self):
        """Reset _time_run_so_far so that all jobs receive new fair allocation
        from here on out.

        Requires self._scheduler_lock to be held when calling this function.
        """

        total_time_run_per_worker_type = {}
        for worker_type in self._worker_types:
            total_time_run_per_worker_type[worker_type] = self._get_total_time_run(worker_type)

        for worker_type in self._worker_types:
            for job_id in self._time_run_so_far:
                if self._num_jobs == 0 or worker_type not in self._time_run_so_far[job_id]:
                    self._time_run_so_far[job_id][worker_type] = 0.0
                else:
                    self._time_run_so_far[job_id][worker_type] -= (
                        total_time_run_per_worker_type[worker_type] / self._num_jobs)
        self._num_jobs = len(self._time_run_so_far)


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _add_to_queue(self, job_id):
        """Adds a job_id to each worker's queue.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        for worker_type in self._worker_types:
            self._per_worker_type_job_queue[worker_type].append([0.0, 0, job_id])


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _remove_from_queue(self, job_id):
        """Removes a job_id from each worker's queue.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' queues.
        """
        for worker_type in self._worker_types:
            for i in range(len(self._per_worker_type_job_queue[worker_type])):
                if self._per_worker_type_job_queue[worker_type][i][2] == job_id:
                    if len(self._per_worker_type_job_queue[worker_type]) > 0:
                        self._per_worker_type_job_queue[worker_type].pop(i)
                    break


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _update_queue(self):
        """Updates each per-worker queue.

        Re-sorts the queue of each worker to compute the next job to run.
        For a given worker w_i, the next job to be scheduled will be the job
        that has so far received the smallest fraction of its computed
        fair allocation.
        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        # Stores the total amount of time run on each worker among currently
        # running jobs.
        tot_time_run = {}

        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            tot_time_run[worker_type] = self._get_total_time_run(worker_type)

        for worker_type in self._worker_types:
            for job_id in self._time_run_so_far:
                if tot_time_run[worker_type] == 0.0:
                    fractions[worker_type][job_id] = 0.0
                else:
                    fraction = self._time_run_so_far[job_id][worker_type] / \
                        tot_time_run[worker_type]
                    fractions[worker_type][job_id] = fraction
            for i in range(len(self._per_worker_type_job_queue[worker_type])):
                [_, _, job_id] = self._per_worker_type_job_queue[worker_type][i]
                self._per_worker_type_job_queue[worker_type][i][0] = fractions[worker_type][job_id] / \
                    self._allocation[job_id][worker_type]
                self._per_worker_type_job_queue[worker_type][i][1] = \
                    self._steps_run_so_far[job_id][worker_type]
            heapq.heapify(self._per_worker_type_job_queue[worker_type])


    def _add_available_worker_id(self, worker_id, timestamp):
        """Adds a worker_id to the list of available workers."""

        self._available_worker_ids.add(timestamp, worker_id)


    def _remove_available_worker_id(self, worker_id=None):
        """Returns the worker_id of the next available worker."""

        if worker_id is None:
            return self._available_worker_ids.remove()
        else:
            return self._available_worker_ids.remove_item(worker_id)


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_highest_priority(self, job_id):
        priorities = []
        for timestamp, worker_id in self._available_worker_ids.queue:
            if timestamp > self._per_job_timestamps.get(job_id, 0):
                continue
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            for i in range(len(self._per_worker_type_job_queue[worker_type])):
                if self._per_worker_type_job_queue[worker_type][i][2] == job_id:
                    priorities.append((self._per_worker_type_job_queue[worker_type][i][0],
                                       worker_id, worker_type))
        priorities.sort(key=lambda x: x[0])
        if len(priorities) == 0:
            return float("inf"), None
        priority = priorities[0][0]
        worker_id = priorities[0][1]
        return priority, worker_id


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_total_steps_run(self, job_id):
        """Returns the total number of steps run for the job with passed-in job_id."""

        # TODO: change to exception
        assert(job_id in self._steps_run_so_far)
        total_steps_run = 0
        for worker_type in self._steps_run_so_far[job_id]:
            total_steps_run += self._steps_run_so_far[job_id][worker_type]
        return total_steps_run


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_total_time_run(self, worker_type):
        """Returns the total time run on the passed-in worker_type since the last reset."""

        total_time_run = 0.0
        for job_id in self._time_run_so_far:
            if worker_type in self._time_run_so_far[job_id]:
                total_time_run += self._time_run_so_far[job_id][worker_type]
        return total_time_run

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(self, worker_type, ip_addr, port, devices):
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
            self._worker_types.add(worker_type)
            self._worker_id_to_worker_type_mapping[worker_id] = worker_type
            self._devices[worker_id] = devices

            if worker_type not in self._per_worker_type_job_queue:
                self._per_worker_type_job_queue[worker_type] = []
                for job_id in self._steps_run_so_far:
                    self._steps_run_so_far[job_id][worker_type] = 0
                    self._throughputs[job_id][worker_type] = \
                        self._compute_throughput(self._jobs[job_id],
                                                 worker_type)
                    # Entries in the queue are sorted by
                    # fraction_run/fraction_allocated, then number of
                    # steps run, then job_id.
                    heapq.heappush(self._per_worker_type_job_queue[worker_type],
                                   [0.0, 0, job_id])

                self._reset_time_run_so_far()

            if self._emulate:
                # For now, assume that all workers are added at timestamp 0 in emulation mode.
                timestamp = 0
            else:
                timestamp = time.time()
            self._add_available_worker_id(worker_id, timestamp)

            self._num_workers += 1
            if not self._emulate:
                self._worker_connections[worker_id] = \
                    scheduler_client.SchedulerRpcClient(ip_addr, port)

            self._allocation = self._get_allocation()

        return worker_id


    def _send_heartbeat_callback(self):
        # TODO.
        pass


    def _done_callback(self, job_id, worker_id, execution_time, num_steps=1,
                       timestamp=None):
        """Handles completion of a scheduled job.

        Updates the running total of completed steps and time spent on each
        worker, for every currently active application. Removes the job from
        the scheduler if the job has finished all its requested steps. Adds
        the worker back to the list of available workers.

        Args:
            job_id: The id of the completed job.
            worker_id: The id of the worker where the job was completed.
            num_steps: The number of steps the job ran for.
        """

        to_remove = None
        with self._scheduler_lock:
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            self._steps_run_so_far[job_id][worker_type] += num_steps
            self._time_run_so_far[job_id][worker_type] += execution_time
            if not self._emulate:
                self._per_job_timestamps[job_id] = time.time()
            print("[Completed] Job ID: %d, Worker ID: %d" % (job_id, worker_id))
            # NOTE: for debug purposes.
            print("[{job_id: {worker_type: steps}}]", self._steps_run_so_far)
            print("[{job_id: {worker_type: time}}]", self._time_run_so_far)
            print()

            if self._get_total_steps_run(job_id) < self._jobs[job_id].num_steps():
                self._add_to_queue(job_id)
            else:
                to_remove = job_id

        if to_remove is not None:
            self.remove_job(job_id)
        if timestamp is None:
            timestamp = time.time()
        self._add_available_worker_id(worker_id, timestamp)
