from __future__ import print_function

import heapq
import numpy as np
import os
import sys
import threading
import time
import datetime
import random

import priority_queue
import utils

SCHEDULER_PORT = 50060
SLEEP_SECONDS = 2
INFINITY = float(1e9)
DEFAULT_THROUGHPUT = INFINITY
DEFAULT_NUM_STEPS = 100     # Default number of steps in each iteration
TIME_PER_ITERATION = 20 * 60    # Time in seconds each iteration should run for
EMA_ALPHA = .25 # Alpha parameter for exponential moving average
MAX_FAILED_ATTEMPTS = 5

np.random.seed(42)
random.seed(42)

class Scheduler:

    class JobIdPair():

        def __init__(self, job0, job1):
            if job0 is None and job1 is None:
                raise ValueError('Cannot form JobIdPair with both ids None')
            elif job0 is None and job1 is not None:
                raise ValueError('First job id in a JobIdPair cannot be None')
            self._job0 = job0
            self._job1 = job1

        def __getitem__(self, index):
            if index == 0:
                return self._job0
            elif index == 1:
                return self._job1
            else:
                raise ValueError('Attempting to access invalid JobIdPair '
                                 'index %d' % index)

        def __lt__(self, other):
            if self[0] != other[0]:
                return self[0] < other[0]
            elif self[1] is None and self[0] is None:
                return False
            elif self[1] is not None and other[1] is not None:
                return self[1] < other[1]
            else:
                return self[1] is None

        def __eq__(self, other):
            return self[0] == other[0] and self[1] == other[1]

        def __hash__(self):
            return hash(self.as_tuple())

        def __repr__(self):
            if self[1] is None:
                return '%d' % (self[0])
            else:
                return ('(%d, %d)' % (self[0], self[1]))

        def as_tuple(self):
            return (self._job0, self._job1)

        def overlaps_with(self, other):
            if self.is_pair():
                raise ValueError('Can only call overlaps_with on a '
                                 'single job id')
            return ((other[0] is not None and self[0] == other[0]) or
                    (other[1] is not None and self[0] == other[1]))

        def is_pair(self):
            return self._job0 is not None and self._job1 is not None

        def singletons(self):
            if self[1] is None:
                return (self,)
            else:
                return (Scheduler.JobIdPair(self[0], None),
                        Scheduler.JobIdPair(self[1], None))

    class JobQueueEntry(object):

        def __init__(self, priority, steps_run, job_id):
            self._priority = priority
            self._steps_run = steps_run
            self._job_id = job_id

        @property
        def priority(self):
            return self._priority

        @priority.setter
        def priority(self, priority):
            self._priority = priority

        @property
        def steps_run(self):
            return self._steps_run

        @steps_run.setter
        def steps_run(self, steps_run):
            self._steps_run = steps_run

        @property
        def job_id(self):
            return self._job_id

        def __lt__(self, other):
            if self._priority != other._priority:
                return self._priority < other._priority
            elif self._steps_run != other._steps_run:
                return self._steps_run < other.steps_run
            else:
                return self._job_id < other.job_id

        def __eq__(self, other):
            return (self._priority == other_.priority
                    and self._steps_run == other._steps_run
                    and self._job_id == other._job_id)

    class JobQueue:

        def __init__(self):
            self._queue = []

        def __getitem__(self, index):
            return self._queue[index]

        def add_job(self, priority, steps_run, job_id, heappush=False):
            entry = Scheduler.JobQueueEntry(priority, steps_run, job_id)
            if heappush:
                heapq.heappush(self._queue, entry)
            else:
                self._queue.append(entry)

        def pop(self, i):
            self._queue.pop(i)

        def heapify(self):
            heapq.heapify(self._queue)

        def update_entry(self, i, priority=None, steps_run=None):
            if priority is not None:
                self._queue[i].priority = priority

            if steps_run is not None:
                self._queue[i].steps_run = steps_run

        def size(self):
            return len(self._queue)

        def get_sorted_queue(self):
            return sorted(self._queue)


    def __init__(self, policy, throughputs_file, job_packing=False):

        # Datastructures to faithfully emulate.
        # Latest emulated timestamp.
        self._current_timestamp = 0
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
        # Policy instance.
        self._policy = policy
        self._job_packing = job_packing
        # RPC clients.
        self._cluster_spec = {}
        self._worker_connections = {}
        # Next job_id to assign.
        self._job_id_counter = 0
        # Next worker_id to assign.
        self._worker_id_counter = 0
        # List of available worker IDs.
        self._available_worker_ids = []
        # Throughputs for all current incomplete applications.
        self._throughputs = {}
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Epochs run on each worker_id, for all current incomplete applications.
        self._steps_run_so_far = {}
        # Time run so far on each worker_id, for all current incomplete
        # applications.
        self._time_run_so_far = {}
        # Cumulative time run so far on each worker_id.
        self._cumulative_time_run_so_far = {}
        # Number of jobs to compute fair share.
        self._num_jobs = 0
        # Commands to run for all current incomplete applications.
        self._jobs = {}
        # Priority queues for each worker_type.
        self._per_worker_type_job_queue = {}
        # Normalizing worker type.
        #self.normalizing_worker_type = normalizing_worker_type
        # The number of steps to run of each job on each worker type
        # for each iteration.
        self._num_steps_per_iteration = {}
        # Throughputs for all job types (pre-measured).
        self._all_throughputs = \
            utils.read_all_throughputs_json(throughputs_file)
        # Verbose flag.
        self._verbose = False

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
        job_id = self.JobIdPair(self._job_id_counter, None)
        self._job_id_counter += 1
        job._job_id = job_id
        self._jobs[job_id] = job
        self._steps_run_so_far[job_id] = {}
        self._time_run_so_far[job_id] = {}
        self._cumulative_time_run_so_far[job_id] = {}
        self._throughputs[job_id] = {}
        self._num_steps_per_iteration[job_id] = {}
        for worker_type in self._worker_types:
            self._steps_run_so_far[job_id][worker_type] = 0
            self._throughputs[job_id][worker_type] = \
                self._compute_throughput(job.job_type, worker_type)
            if self._job_packing:
                self._populate_job_combination_metadata(job_id,
                                                        worker_type)
            if self._policy.name == 'FIFO':
                self._num_steps_per_iteration[job_id][worker_type] = \
                        self._jobs[job_id].total_steps
            else:
                throughput = self._throughputs[job_id][worker_type]
                self._num_steps_per_iteration[job_id][worker_type] = \
                    int(throughput * TIME_PER_ITERATION)
            self._cumulative_time_run_so_far[job_id][worker_type] = 0.0

        self._reset_time_run_so_far()
        self._add_to_queue(job_id)
        self._allocation = self._get_allocation()
        self._per_job_start_timestamps[job_id] = self._current_timestamp
        print('%s] [Job dispatched] Job ID: %s' % (self._current_timestamp,
                                                   str(job_id)))
        return job_id


    def remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        job_id = self.JobIdPair(job_id, None)
        duration = self._per_job_latest_timestamps[job_id] - \
            self._per_job_start_timestamps[job_id]
        self._job_completion_times[job_id] = \
                (duration, self._jobs[job_id].duration)
        print("Job %d completed\n\tStart timestamp: %.2f\n\t"
              "End timestamp: %.2f\nDuration: %.2f %s\n" % (
                  job_id[0],
                  self._per_job_start_timestamps[job_id],
                  self._per_job_latest_timestamps[job_id],
                  duration, "seconds")
              )

        self._reset_time_run_so_far()

        del self._jobs[job_id]
        del self._steps_run_so_far[job_id]
        del self._time_run_so_far[job_id]
        del self._cumulative_time_run_so_far[job_id]
        del self._throughputs[job_id]
        if self._job_packing:
            to_delete = []
            for other_job_id in self._throughputs:
                if (other_job_id.is_pair() and
                    job_id.overlaps_with(other_job_id)):
                    for only_other_job_id in other_job_id.singletons():
                        if only_other_job_id != job_id:
                            for worker_type in self._worker_types:
                                self._steps_run_so_far[only_other_job_id][worker_type] += \
                                        self._steps_run_so_far[other_job_id][worker_type]
                    to_delete.append(other_job_id)
            for other_job_id in to_delete:
                del self._throughputs[other_job_id]
                del self._steps_run_so_far[other_job_id]
                del self._time_run_so_far[other_job_id]

        self._remove_from_queue(job_id)
        if len(self._throughputs) > 0:
            self._allocation = self._get_allocation()


    def num_workers(self):
        """Returns the number of workers the scheduler is connected to."""

        n = 0
        with self._scheduler_lock:
            for worker_type in self._cluster_spec:
                n += self._cluster_spec[worker_type]
            return n


    def is_done(self):
        """Returns whether the scheduler is done with all its assigned work."""
        with self._scheduler_lock:
            return (len(self._event_queue) == 0 and
                    len(self._steps_run_so_far) == 0)

    def pending_jobs(self):
        """Returns whether there are any outstanding jobs to schedule."""
        with self._scheduler_lock:
            return len(self._jobs) > 0

    def shutdown(self, logfile=None):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        output = []
        with self._scheduler_lock:
            if len(self._job_completion_times) == 0:
                return
            output.append("Job completion times:"
                          "\n\t%s" % self._job_completion_times)
            average_job_completion_time = \
                sum([x[0] for x in self._job_completion_times.values()]) / \
                len(self._job_completion_times)
            unit = "seconds"
            output.append("Average job completion time: "
                          "%.3f %s" % (average_job_completion_time, unit))
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
                    f.write(line + '\n')
        # TODO: Any other cleanup?
        sys.exit(0)


    """
    ======================================================================
       Scheduler's main _schedule() method.
    ======================================================================
    """

    def emulate(self, cluster_spec, arrival_times, jobs):
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

        remaining_jobs = len(jobs)
        queued_jobs = []
        running_jobs = []

        # Set up the cluster according to the provided spec
        worker_types = sorted([worker_type for worker_type in cluster_spec])
        for worker_type in worker_types:
            for i in range(cluster_spec[worker_type]):
                self._register_worker_callback(worker_type)

        # Add all jobs to the queue
        for i in range(1, len(arrival_times)):
            assert(arrival_times[i] >= arrival_times[i-1])

        for (arrival_time, job) in zip(arrival_times, jobs):
            queued_jobs.append((arrival_time, job))

        while remaining_jobs > 0:
            # Jump to the next event's timestamp
            min_timestamp = INFINITY
            if len(running_jobs) > 0 and running_jobs[0][0] < min_timestamp:
                min_timestamp = running_jobs[0][0]
            if len(queued_jobs) > 0 and queued_jobs[0][0] < min_timestamp:
                min_timestamp = queued_jobs[0][0]
            if min_timestamp is not INFINITY:
                self._current_timestamp = min_timestamp

            # Check if any jobs have completed
            while len(running_jobs) > 0:
                (finish_time, job_id, worker_id, num_steps) = running_jobs[0]
                if finish_time <= self._current_timestamp:
                    self._done_callback(job_id[0], worker_id, num_steps)
                    if job_id not in self._jobs:
                        remaining_jobs -= 1
                    heapq.heappop(running_jobs)
                else:
                    break

            # Dispatch any newly arrived jobs
            while len(queued_jobs) > 0:
                (arrival_time, job) = queued_jobs[0]
                if arrival_time <= self._current_timestamp:
                    job_id = self.add_job(job)
                    queued_jobs.pop(0)
                else:
                    break

            # Schedule jobs until there are no available workers or no jobs
            # with non-zero allocations on available workers
            seen_worker_ids = set()
            while True:
                worker_id = self._remove_available_worker_id()
                if worker_id in seen_worker_ids:
                    self._add_available_worker_id(worker_id)
                    break
                elif worker_id is None:
                    break
                else:
                    seen_worker_ids.add(worker_id)
                worker_type = self._worker_id_to_worker_type_mapping[worker_id]
                self._update_queue()
                if self._per_worker_type_job_queue[worker_type].size() == 0:
                    self._add_available_worker_id(worker_id)
                    continue
                queued_job = self._per_worker_type_job_queue[worker_type][0]
                job_id = queued_job.job_id
                priority = queued_job.priority

                # If the chosen job has an allocation of zero, return the worker
                # to the available worker pool.
                if self._allocation[job_id][worker_type] == 0.0:
                    # Move worker_id to the end of the queue.
                    self._add_available_worker_id(worker_id)
                    continue

                for single_job_id in job_id.singletons():
                    self._remove_from_queue(single_job_id)
                # self._print_allocation()
                for single_job_id in job_id.singletons():
                    self._num_steps_per_iteration[single_job_id][worker_type] = \
                        min(self._num_steps_per_iteration[single_job_id][worker_type],
                            self._get_remaining_steps(single_job_id))
                    num_steps = \
                        self._num_steps_per_iteration[single_job_id][worker_type]
                    if num_steps <= 0:
                        raise ValueError('Num steps should be greater than 0, is %d' % (num_steps))
                    print('Running job %s for %d steps' % (job_id, num_steps))
                    print(('%s] [Micro-task scheduled] Job ID: %s, '
                           'Worker type: %s') % (self._current_timestamp,
                                                 job_id, worker_type))

                    finish_time = (self._current_timestamp +
                                   (num_steps /
                                    self._throughputs[job_id][worker_type]))
                    heapq.heappush(running_jobs, (finish_time, job_id,
                                                  worker_id, num_steps))
                    self._per_job_latest_timestamps[job_id] = \
                        self._current_timestamp
        print('Total duration: %.3f seconds' % (self._current_timestamp))
    """
    ======================================================================
       Helper methods to compute each user's fair allocation.
    ======================================================================
    """

    def _print_allocation(self):
        """Prints the allocation."""
        print('[DEBUG] Allocation:')
        job_ids = sorted([job_id for job_id in self._allocation])
        for job_id in job_ids:
            line = 'Job %s: ' % (job_id)
            for worker_type in self._allocation[job_id]:
                line += '[%4s %.3f] ' % (worker_type, self._allocation[job_id][worker_type])
            print(line)

    def _get_allocation(self):
        """Computes the allocation.

        Uses the specified policy to compute an allocation of jobs to
        compute resources. Requires self._scheduler_lock to be held
        when calling this function.

        Returns:
            A 2-level dict indexed by job_id and then worker_type. For
            example,

            {0: {"v100": 0.25, "p100": 0.95}, 1: {"v100": 0.75, "p100": 0.05}}

            indicates that for 25% of the time, worker type 'v100' should run,
            job 0 and for 95% of the time, worker type 'p100' should run job 0.
        """

        unflattened_allocation = self._policy.get_allocation(self._throughputs,
                                                             self._cluster_spec)
        if self._verbose:
            print("New allocation\n\t%s\n" % unflattened_allocation)
        return unflattened_allocation


    def _populate_job_combination_metadata(self, job_id, worker_type):
        """Populate metadata for job combinations involving passed-in job_id."""

        job = self._jobs[job_id]
        for other_job_id in self._jobs:
            if other_job_id != job_id:
                other_job = self._jobs[other_job_id]
                merged_job_id = self.JobIdPair(job_id[0], other_job_id[0])
                if merged_job_id not in self._throughputs:
                    self._throughputs[merged_job_id] = {}
                    self._steps_run_so_far[merged_job_id] = {}
                    self._time_run_so_far[merged_job_id] = {}
                    self._cumulative_time_run_so_far[merged_job_id] = {}
                self._throughputs[merged_job_id][worker_type] = \
                    self._compute_throughput([job, other_job], worker_type)
                self._steps_run_so_far[merged_job_id][worker_type] = 0


    def _compute_throughput(self, job_type, worker_type):
        return self._all_throughputs[job_type][worker_type]

    """
    ======================================================================
       Methods to update the scheduler's internal data structures.
    ======================================================================
    """


    def _reset_time_run_so_far(self):
        """Reset _time_run_so_far so that all jobs receive new fair allocation
        from here on out.

        Requires self._scheduler_lock to be held when calling this function.
        """
        """
        job_ids = sorted([job_id for job_id in self._time_run_so_far])
        for job_id in job_ids:
            time_line = '[DEBUG_ALLOCATION]\tJob %s time run so far:\t' % (job_id)
            allocation_line = '[DEBUG_ALLOCATION]\tJob %s allocation:\t' % (job_id)
            worker_types = \
                    sorted([worker_type for worker_type in self._time_run_so_far[job_id]])
            for worker_type in worker_types:
                allocation_line += '[%4s %.3f]\t' % (worker_type, self._allocation[job_id][worker_type])
                time_line += '[%4s %.3f]\t' % (worker_type, self._time_run_so_far[job_id][worker_type])
            print(allocation_line)
            print(time_line)
        """
        for worker_type in self._worker_types:
            for job_id in self._time_run_so_far:
                self._time_run_so_far[job_id][worker_type] = 0.0
        self._num_jobs = len(self._time_run_so_far)

    def _add_to_queue(self, job_id):
        """Adds a job_id to each worker's queue.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        for worker_type in self._worker_types:
            self._per_worker_type_job_queue[worker_type].add_job(0.0, 0,
                                                                 job_id)
            for other_job_id in self._throughputs:
                if (other_job_id.is_pair() and
                    job_id.overlaps_with(other_job_id)):
                    self._per_worker_type_job_queue[worker_type].add_job(
                                0.0, 0.0, other_job_id)


    def _remove_from_queue(self, job_id):
        """Removes a job_id from each worker's queue.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' queues.
        """
        for worker_type in self._worker_types:
            while True:
                found = False
                for i in range(self._per_worker_type_job_queue[worker_type].size()):
                    queued_job = self._per_worker_type_job_queue[worker_type][i]
                    if job_id.overlaps_with(queued_job.job_id):
                        if self._per_worker_type_job_queue[worker_type].size() > 0:
                            self._per_worker_type_job_queue[worker_type].pop(i)
                            found = True
                        break
                if not found:
                    break


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
                    fraction = 1.0 / len(self._worker_types)
                else:
                    fraction = self._time_run_so_far[job_id][worker_type] / \
                        tot_time_run[worker_type]
                fractions[worker_type][job_id] = fraction
            for i in range(self._per_worker_type_job_queue[worker_type].size()):
                queued_job = self._per_worker_type_job_queue[worker_type][i]
                job_id = queued_job.job_id
                if self._allocation[job_id][worker_type] == 0.0:
                    self._per_worker_type_job_queue[worker_type].update_entry(
                            i, priority=float("inf"))
                else:
                    new_priority = fractions[worker_type][job_id] /\
                            self._allocation[job_id][worker_type]
                    steps_run = self._steps_run_so_far[job_id][worker_type]
                    self._per_worker_type_job_queue[worker_type].update_entry(
                            i, priority=new_priority, steps_run=steps_run)
            self._per_worker_type_job_queue[worker_type].heapify()


    def _add_available_worker_id(self, worker_id):
        """Adds a worker_id to the list of available workers."""

        self._available_worker_ids.append(worker_id)


    def _remove_available_worker_id(self, worker_id=None):
        """Returns the worker_id of the next available worker."""

        if len(self._available_worker_ids) == 0:
            return None
        return self._available_worker_ids.pop(0)


    def _get_highest_priority(self, job_id):
        priorities = []
        for timestamp, worker_id in self._available_worker_ids.queue:
            if timestamp > self._per_job_latest_timestamps.get(job_id, 0):
                continue
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            for i in range(self._per_worker_type_job_queue[worker_type].size()):
                queued_job = self._per_worker_type_job_queue[worker_type][i]
                if queued_job.job_id == job_id:
                    priorities.append((queued_job.priority, worker_id,
                                       worker_type))
        priorities.sort(key=lambda x: x[0])
        if len(priorities) == 0:
            return float("inf"), None
        priority = priorities[0][0]
        worker_id = priorities[0][1]
        return priority, worker_id


    def _get_total_steps_run(self, job_id):
        """Returns the total number of steps run for job with id job_id."""

        # TODO: change to exception
        assert(job_id in self._steps_run_so_far)
        total_steps_run = 0
        for worker_type in self._steps_run_so_far[job_id]:
            total_steps_run += self._steps_run_so_far[job_id][worker_type]
        for other_job_id in self._steps_run_so_far:
            if other_job_id.is_pair() and job_id.overlaps_with(other_job_id):
                for worker_type in self._steps_run_so_far[other_job_id]:
                    total_steps_run += \
                            self._steps_run_so_far[other_job_id][worker_type]
        return total_steps_run


    def _get_remaining_steps(self, job_id):
        steps_run_so_far = self._get_total_steps_run(job_id)
        return self._jobs[job_id].total_steps - steps_run_so_far

    def _get_total_time_run(self, worker_type):
        """Returns the total time run on worker_type since the last reset."""

        total_time_run = 0.0
        for job_id in self._time_run_so_far:
            if worker_type in self._time_run_so_far[job_id]:
                total_time_run += self._time_run_so_far[job_id][worker_type]
        return total_time_run

    def _get_total_time_run_for_job(self, job_id):
        """Returns the total time run for job with id job_id."""
        total_time_run = 0.0
        for worker_type in self._cumulative_time_run_so_far[job_id]:
            total_time_run += \
                    self._cumulative_time_run_so_far[job_id][worker_type]
        return total_time_run

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(self, worker_type):
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

        worker_id = self._worker_id_counter
        self._worker_ids.append(worker_id)
        self._worker_id_counter += 1
        self._worker_types.add(worker_type)
        self._worker_id_to_worker_type_mapping[worker_id] = worker_type

        if worker_type not in self._per_worker_type_job_queue:
            self._per_worker_type_job_queue[worker_type] = self.JobQueue()
            for job_id in self._jobs:
                self._steps_run_so_far[job_id][worker_type] = 0
                self._time_run_so_far[job_id][worker_type] = 0
                self._cumulative_time_run_so_far[job_id][worker_type] = 0.0
                self._throughputs[job_id][worker_type] = \
                    self._compute_throughput(self._jobs[job_id],
                                             worker_type)

                if self._job_packing:
                    self._populate_job_combination_metadata(job_id,
                                                            worker_type)
                if self._policy.name == 'FIFO':
                    self._num_steps_per_iteration[job_id][worker_type] = \
                            self._total_steps
                else:
                    self._num_steps_per_iteration[job_id][worker_type] = \
                        (self._throughputs[job_id][worker_type] *
                            TIME_PER_ITERATION)
                # Entries in the queue are sorted by
                # fraction_run/fraction_allocated, then number of
                # steps run, then job_id.
                self._per_worker_type_job_queue[worker_type].add_job(
                        0.0, 0, job_id, heappush=True)

        self._reset_time_run_so_far()
        self._add_available_worker_id(worker_id)
        if worker_type not in self._cluster_spec:
            self._cluster_spec[worker_type] = 0
        self._cluster_spec[worker_type] += 1
        self._allocation = self._get_allocation()

        return worker_id


    def _done_callback(self, job_id, worker_id, num_steps):
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
        job_id = self.JobIdPair(job_id, None)

        to_remove = []

        worker_type = self._worker_id_to_worker_type_mapping[worker_id]

        start_timestamp = self._per_job_latest_timestamps[job_id]
        execution_time = self._current_timestamp - start_timestamp

        self._steps_run_so_far[job_id][worker_type] += num_steps
        self._time_run_so_far[job_id][worker_type] += execution_time
        self._cumulative_time_run_so_far[job_id][worker_type] += execution_time

        for single_job_id in job_id.singletons():
            self._per_job_latest_timestamps[single_job_id] = \
                self._current_timestamp
            print(('%s] [Micro-task succeeded] '
                   'Job ID: %s, Worker type: %s') % (self._current_timestamp,
                                                   job_id, worker_type))
            # NOTE: for debug purposes.
            if self._verbose:
                print("[{job_id: {worker_type: steps}}]",
                      self._steps_run_so_far)
                print("[{job_id: {worker_type: time}}]", self._time_run_so_far)
                print()

        for single_job_id in job_id.singletons():
            if (self._get_total_steps_run(single_job_id) <
                self._jobs[single_job_id].total_steps):
                self._add_to_queue(single_job_id)
            else:
                print(('%s] [Job succeeded] '
                       'Job ID: %s') % (self._current_timestamp,
                                        single_job_id))
                to_remove.append(single_job_id)

        self._add_available_worker_id(worker_id)

        for single_job_id in to_remove:
            self.remove_job(single_job_id[0])
