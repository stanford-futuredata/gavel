from __future__ import print_function

import heapq
import numpy as np
import os
from preconditions import preconditions
import sys
import threading
import time

import priority_queue
from runtime.rpc import scheduler_server, scheduler_client
import utils

SCHEDULER_PORT = 50060
SLEEP_SECONDS = 2

class Scheduler:

    def __init__(self, policy, get_num_steps_to_run, emulate=False,
                 normalizing_worker_type=None, throughputs_directory=None,
                 job_packing=False):
        # Emulate flag.
        self._emulate = emulate

        # Datastructures to faithfully emulate.
        # Latest emulated timestamp.
        self._timestamp = 0
        # Start and last processed timestamp for each job_id.
        self._per_job_start_timestamps = {}
        self._per_job_latest_timestamps = {}
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
        self._job_packing = job_packing
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
        # Verbose flag.
        self._verbose = False

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
                    return
                # If passed-in timestamp is before the first timestamp in the event queue,
                # and no jobs are scheduled to run.
                if timestamp < self._event_queue[0][0] and len(self._steps_run_so_far) > 0:
                    return
                (_, func, args) = self._event_queue.pop(0)
            func(*args, timestamp=timestamp)

    """
    ======================================================================
       Public-facing scheduler methods.
    ======================================================================
    """

    def add_job(self, job, timestamp=None):
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
            if self._emulate:
                assert timestamp is not None
            else:
                timestamp = time.time()

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
                if self._job_packing:
                    self._population_job_combination_metadata(job_id, worker_type)

            self._reset_time_run_so_far(timestamp)
            self._add_to_queue(job_id)
            self._allocation = self._get_allocation()
            if self._emulate:
                self._per_job_start_timestamps[job_id] = timestamp
            else:
                self._per_job_start_timestamps[job_id] = time.time()
        return job_id


    def remove_job(self, job_id, timestamp=None):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        with self._scheduler_lock:
            duration = self._per_job_latest_timestamps[job_id] - \
                self._per_job_start_timestamps[job_id]
            self._job_completion_times[job_id] = (duration,
                                                  self._jobs[job_id].duration())
            print("Job %d completed\n\tStart timestamp: %.2f\n\t"
                  "End timestamp: %.2f\nDuration: %.2f %s\n" % (
                      job_id,
                      self._per_job_start_timestamps[job_id],
                      self._per_job_latest_timestamps[job_id],
                      duration,
                      "timeunits" if self._emulate else "seconds")
                  )

            if self._emulate:
                assert timestamp is not None
            else:
                timestamp = time.time()
            self._reset_time_run_so_far(timestamp)

            del self._jobs[job_id]
            del self._steps_run_so_far[job_id]
            del self._time_run_so_far[job_id]
            del self._throughputs[job_id]
            if self._job_packing:
                to_delete = []
                for job_id_combination in self._throughputs:
                    if isinstance(job_id_combination, tuple):
                        if job_id in job_id_combination:
                            for other_job_id in job_id_combination:
                                if other_job_id != job_id:
                                    for worker_type in self._worker_types:
                                        self._steps_run_so_far[other_job_id][worker_type] +=\
                                            self._steps_run_so_far[job_id_combination][worker_type]
                            to_delete.append(job_id_combination)
                for job_id_combination in to_delete:
                    del self._throughputs[job_id_combination]
                    del self._steps_run_so_far[job_id_combination]
                    del self._time_run_so_far[job_id_combination]

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


    def shutdown(self, logfile=None):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        output = []
        with self._scheduler_lock:
            if self._emulate:
                output.append("Total time taken: %.2f timeunits" % self._timestamp)
            output.append("Job completion times:\n\t%s" % self._job_completion_times)
            average_job_completion_time = \
                sum([x[0] for x in self._job_completion_times.values()]) / \
                len(self._job_completion_times)
            unit = "timeunits" if self._emulate else "seconds"
            output.append("Average job completion time: %.3f %s" % (average_job_completion_time,
                                                                    unit))
            for worker_id in self._worker_connections:
                self._worker_connections[worker_id].shutdown()
        if logfile is None:
            for line in output:
                print(line)
        else:
            if logfile[0] != '/':
                path = os.path.join(os.getcwd(), logfile)
            else:
                path = logfile
            with open(path, 'w') as f:
                for line in output:
                    f.write(line)
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
                self.execute_from_event_queue(timestamp)
            with self._scheduler_lock:
                worker_type = self._worker_id_to_worker_type_mapping[worker_id]
                self._update_queue()
                if len(self._per_worker_type_job_queue[worker_type]) == 0:
                    if not self._emulate:
                        timestamp = time.time()
                    self._add_available_worker_id(worker_id, timestamp)
                    continue

                # TODO: Change job_id_combination to be a tuple always.
                # job_id_combination is a tuple if multiple applications running;
                # otherwise just an integer.
                [priority, _, job_id_combination] = \
                    self._per_worker_type_job_queue[worker_type][0]
                if len(job_id_combination) == 1:
                    job_id_combination = job_id_combination[0]

                # If the highest priority job involves waiting, pick an earlier job
                # with reasonably high priority.
                latest_timestamp = 0
                if isinstance(job_id_combination, tuple):
                    for job_id in job_id_combination:
                        latest_timestamp = max(latest_timestamp,
                                               self._per_job_latest_timestamps.get(job_id, 0))
                else:
                    latest_timestamp = self._per_job_latest_timestamps.get(
                        job_id_combination, 0)
                if timestamp < latest_timestamp:
                    found_jobs = []
                    sorted_queue = sorted(self._per_worker_type_job_queue[worker_type])
                    for i in range(1, len(sorted_queue)):
                        [ready_priority, _, ready_job_id_combination] = sorted_queue[i]
                        latest_ready_timestamp = 0
                        for ready_job_id in ready_job_id_combination:
                            latest_ready_timestamp = max(
                                latest_ready_timestamp,
                                self._per_job_latest_timestamps.get(ready_job_id, 0))
                        if latest_timestamp > latest_ready_timestamp:
                            found_jobs.append((latest_ready_timestamp, ready_priority,
                                               ready_job_id_combination))
                            break
                    # Sort by the timestamp when ready.
                    found_jobs.sort()
                    if len(found_jobs) > 0:
                        # Swap jobs, so that the next job scheduled is available faster.
                        (_, ready_priority, ready_job_id_combination) = found_jobs[0]
                        # Pick a job_combination with up to 3x higher priority if
                        # the job_combination can be scheduled earlier.
                        if ready_priority < (3 * priority):
                            priority = ready_priority
                            job_id_combination = ready_job_id_combination
                            if len(job_id_combination) == 1:
                                job_id_combination = job_id_combination[0]

                # If the chosen job has an allocation of zero, return the worker to
                # the available worker pool.
                if self._allocation[job_id_combination][worker_type] == 0.0:
                    if not self._emulate:
                        timestamp = time.time()
                    all_timestamps = self._available_worker_ids.get_unique_keys_sorted()
                    if self._emulate:
                        if len(all_timestamps) > 1:
                            timestamp = all_timestamps[1]
                    # Move worker_id to behind the second worker.
                    self._add_available_worker_id(worker_id, timestamp+0.1)
                    continue

                # Get available worker_id with the highest priority for this particular job_id.
                highest_priority, worker_id_with_highest_priority = \
                    self._get_highest_priority(job_id_combination)
                if priority > highest_priority:  # Lower is better.
                    timestamp_with_highest_priority, worker_id_with_highest_priority = \
                        self._remove_available_worker_id(worker_id=worker_id_with_highest_priority)
                    # If removal succeeded.
                    if worker_id_with_highest_priority is not None:
                        self._add_available_worker_id(worker_id, timestamp)
                        worker_id = worker_id_with_highest_priority
                        timestamp = timestamp_with_highest_priority
                        worker_type = self._worker_id_to_worker_type_mapping[worker_id]

                if isinstance(job_id_combination, tuple):
                    for job_id in job_id_combination:
                        timestamp = max(
                            timestamp, self._per_job_latest_timestamps.get(job_id, 0))
                        self._remove_from_queue(job_id)
                else:
                    timestamp = max(
                        timestamp,self._per_job_latest_timestamps.get(job_id_combination, 0))
                    self._remove_from_queue(job_id_combination)

                # Actually execute the scheduled job_id_combination on the right worker_id.
                if not self._emulate:
                    # TODO: Do something with job_id_combination.
                    if isinstance(job_id_combination, tuple):
                        for job_id in job_id_combination:
                            num_steps = self._get_num_steps_to_run(job_id, worker_type)
                            self._worker_connections[worker_id].run([(job_id,
                                                                      self._jobs[job_id].command(),
                                                                      num_steps)])
                    else:
                        num_steps = self._get_num_steps_to_run(job_id_combination,
                                                               worker_type)
                        self._worker_connections[worker_id].run([(job_id_combination,
                                                                  self._jobs[job_id_combination].command(),
                                                                  num_steps)])

            # Can only call _done_callback with lock released.
            if self._emulate:
                # When emulating, directly call _done_callback since there's no worker.
                if not isinstance(job_id_combination, tuple):
                    throughputs = (self._throughputs[job_id_combination][worker_type],)
                    job_id_combination = (job_id_combination,)
                else:
                    throughputs = self._throughputs[job_id_combination][worker_type]
                durations = []
                for job_id, throughput in zip(job_id_combination, throughputs):
                    duration = self._jobs[job_id].duration()
                    # TODO: change to exception.
                    assert duration is not None
                    if self.normalizing_worker_type is not None:
                        normalizing_factor = throughput / \
                            self._throughputs[job_id][self.normalizing_worker_type]
                        duration /= normalizing_factor
                    durations.append(duration)
                for (job_id, duration) in zip(job_id_combination, durations):
                    print("[Job ID: %d, Worker ID: %d [%s]] Start: %d, End: %d" % (
                        job_id, worker_id, worker_type, timestamp, timestamp+duration))
                # TODO: Can do more fine-grained accounting for duration here.
                self._done_callback(job_id_combination, worker_id,
                                    max(durations),
                                    timestamp=timestamp+duration)
                self._timestamp = max(self._timestamp, timestamp+max(durations))

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

        unflattened_allocation = self._policy.get_allocation(
            self._throughputs)
        if self._verbose:
            print("New allocation\n\t%s\n" % unflattened_allocation)
        return unflattened_allocation


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _population_job_combination_metadata(self, job_id, worker_type):
        """Populate metadata for job combinations involving passed-in job_id."""

        job = self._jobs[job_id]
        for other_job_id in self._jobs:
            if other_job_id != job_id:
                other_job = self._jobs[other_job_id]
                if (job_id, other_job_id) not in self._throughputs:
                    self._throughputs[(job_id, other_job_id)] = {}
                    self._steps_run_so_far[(job_id, other_job_id)] = {}
                    self._time_run_so_far[(job_id, other_job_id)] = {}
                self._throughputs[(job_id, other_job_id)][worker_type] = \
                    self._compute_throughput([job, other_job], worker_type)
                self._steps_run_so_far[(job_id, other_job_id)][worker_type] = 0


    def _compute_throughput(self, jobs, worker_type, other_jobs=None):
        import itertools
        if not isinstance(jobs, list):
            job = jobs
            job_type = job.job_type()
            job_type = tuple([job_type])
            if job_type in self._all_throughputs and worker_type in self._all_throughputs[job_type]:
                throughput = self._all_throughputs[job_type][worker_type]
                return throughput[0]
        else:
            job_types = []
            for job in jobs:
                job_types.append(job.job_type())
            for permutation in itertools.permutations(job_types):
                permutation = tuple(permutation)
                if permutation in self._all_throughputs and worker_type in self._all_throughputs[permutation]:
                    throughputs = self._all_throughputs[permutation][worker_type]
                    throughputs_dict = {}
                    for elem, throughput in zip(permutation, throughputs):
                        throughputs_dict[elem] = throughput
                    return tuple([throughputs_dict[elem] for elem in job_types])
        # TODO: compute throughput.
        # TODO: add parameter for device_id?
        return 10

    """
    ======================================================================
       Methods to update the scheduler's internal data structures.
    ======================================================================
    """


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _reset_time_run_so_far(self, timestamp):
        """Reset _time_run_so_far so that all jobs receive new fair allocation
        from here on out.

        Requires self._scheduler_lock to be held when calling this function.
        """

        for worker_type in self._worker_types:
            for job_id_combination in self._time_run_so_far:
                self._time_run_so_far[job_id_combination][worker_type] = 0.0
        self._num_jobs = len(self._time_run_so_far)


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _add_to_queue(self, job_id):
        """Adds a job_id to each worker's queue.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        for worker_type in self._worker_types:
            self._per_worker_type_job_queue[worker_type].append([0.0, 0, (job_id,)])
            for job_id_combination in self._throughputs:
                if isinstance(job_id_combination, tuple):
                    if job_id in job_id_combination:
                        self._per_worker_type_job_queue[worker_type].append(
                            [0.0, 0.0, job_id_combination])


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _remove_from_queue(self, job_id):
        """Removes a job_id from each worker's queue.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' queues.
        """
        for worker_type in self._worker_types:
            while True:
                found = False
                for i in range(len(self._per_worker_type_job_queue[worker_type])):
                    job_id_combination = self._per_worker_type_job_queue[worker_type][i][2]
                    if job_id in job_id_combination:
                        if len(self._per_worker_type_job_queue[worker_type]) > 0:
                            self._per_worker_type_job_queue[worker_type].pop(i)
                            found = True
                        break
                if not found:
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
                    fractions[worker_type][job_id] = 1.0 / len(self._worker_types)
                else:
                    fraction = self._time_run_so_far[job_id][worker_type] / \
                        tot_time_run[worker_type]
                    fractions[worker_type][job_id] = fraction
            for i in range(len(self._per_worker_type_job_queue[worker_type])):
                [_, _, job_id] = self._per_worker_type_job_queue[worker_type][i]
                if len(job_id) == 1:
                    job_id = job_id[0]
                if self._allocation[job_id][worker_type] == 0.0:
                    self._per_worker_type_job_queue[worker_type][i][0] = float("inf")
                else:
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

    def _update_available_worker_id_keys(self, new_timestamp):
        """Updates the timestamp field in the list of available workers."""

        self._available_worker_ids.update_key(new_timestamp)


    @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_highest_priority(self, job_id):
        priorities = []
        for timestamp, worker_id in self._available_worker_ids.queue:
            if timestamp > self._per_job_latest_timestamps.get(job_id, 0):
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
        for job_id_combination in self._steps_run_so_far:
            if isinstance(job_id_combination, tuple) and job_id in job_id_combination:
                for worker_type in self._steps_run_so_far[job_id_combination]:
                    total_steps_run += self._steps_run_so_far[job_id_combination][worker_type]
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

    def _register_worker_callback(self, worker_type, ip_addr, port, devices,
                                  timestamp=None):
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
                for job_id in self._jobs:
                    self._steps_run_so_far[job_id][worker_type] = 0
                    self._throughputs[job_id][worker_type] = \
                        self._compute_throughput(self._jobs[job_id],
                                                 worker_type)

                    if self._job_packing:
                        self._population_job_combination_metadata(job_id, worker_type)

                    # Entries in the queue are sorted by
                    # fraction_run/fraction_allocated, then number of
                    # steps run, then job_id.
                    heapq.heappush(self._per_worker_type_job_queue[worker_type],
                                   [0.0, 0, (job_id,)])

                if self._emulate:
                    assert(timestamp is not None)
                else:
                    timestamp = time.time()
                self._reset_time_run_so_far(timestamp)

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


    def _done_callback(self, job_id_combination, worker_id, execution_time, num_steps=1,
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

        to_remove = []
        with self._scheduler_lock:
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            if not isinstance(job_id_combination, tuple):
                job_id_combination = (job_id_combination,)
            # TODO: Fix this, pretty messy!
            if len(job_id_combination) == 1:
                self._steps_run_so_far[job_id_combination[0]][worker_type] += num_steps
                self._time_run_so_far[job_id_combination[0]][worker_type] += execution_time
            else:
                self._steps_run_so_far[job_id_combination][worker_type] += num_steps
                self._time_run_so_far[job_id_combination][worker_type] += execution_time
            for job_id in job_id_combination:
                if not self._emulate:
                    self._per_job_latest_timestamps[job_id] = time.time()
                else:
                    self._per_job_latest_timestamps[job_id] = timestamp
            print("[Completed] Job ID: %s, Worker ID: %d" % (job_id_combination, worker_id))
            # NOTE: for debug purposes.
            if self._verbose:
                print("[{job_id: {worker_type: steps}}]", self._steps_run_so_far)
                print("[{job_id: {worker_type: time}}]", self._time_run_so_far)
                print()

            for job_id in job_id_combination:
                if self._get_total_steps_run(job_id) < self._jobs[job_id].num_steps():
                    self._add_to_queue(job_id)
                else:
                    to_remove.append(job_id)

            if timestamp is None:
                timestamp = time.time()
            self._add_available_worker_id(worker_id, timestamp)

            if len(self._per_job_latest_timestamps) > 0:
                self._update_available_worker_id_keys(
                    min(self._per_job_latest_timestamps.values()))

        for job_id in to_remove:
            self.remove_job(job_id, timestamp=timestamp)
