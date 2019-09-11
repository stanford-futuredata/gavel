from __future__ import print_function

import heapq
import numpy as np
import os
# from preconditions import preconditions
import queue
import sys
import threading
import time
import datetime
import random
import math
import matrix_completion
import warnings
import sys

# TODO: clean these up.
from job import Job
import job_id_pair
from job_table import JobTable
from runtime.rpc import scheduler_server, scheduler_client
import utils

SCHEDULER_PORT = 50060
SLEEP_SECONDS = 2
INFINITY = float("inf")
DEFAULT_THROUGHPUT = INFINITY
DEFAULT_NUM_STEPS = 100     # Default number of steps in each iteration.
EMA_ALPHA = .25 # Alpha parameter for exponential moving average.
MAX_FAILED_ATTEMPTS = 5
DEFAULT_MATRIX_COMPLETION_K = 10
DEFAULT_MATRIX_COMPLETION_MU = 1e-2

class Scheduler:

    def __init__(self, policy, simulate=False, throughputs_file=None,
                 seed=0, time_per_iteration=1920, profiling_percentage=0.0,
                 num_reference_models=26):

        # Scheduling occurs in rounds.
        print('Running scheduler with policy=%s, schedule_in_rounds=True, '
               'seed=%d, time_per_iteration=%d, '
               'profiling_percentage=%f' % (policy.name,
                                            seed,
                                            time_per_iteration,
                                            profiling_percentage))

        self._num_correct_predictions = 0
        # Flag to control whether scheduler runs in simulation mode.
        self._simulate = simulate
        # Initialize seeds.
        self._initialize_seeds(seed)
        # Initialize time in seconds each iteration should run for.
        self._time_per_iteration = time_per_iteration

        # Latest simulated timestamp.
        self._current_timestamp = 0
        # Start and last processed timestamp for each job_id.
        self._per_job_start_timestamps = {}
        self._per_job_latest_timestamps = {}
        # Job completion times.
        self._job_completion_times = {}
        # Job priority weights.
        self._job_priority_weights = {}
        # Queue of events that need to be processed at specific timestamps.
        self._event_queue = []

        # List of worker IDs.
        self._worker_ids = []
        # List of worker types.
        self._worker_types = set()
        # Mapping of worker ID to worker type, and worker type to worker ID.
        self._worker_id_to_worker_type_mapping = {}
        self._worker_type_to_worker_id_mapping = {}
        # Policy instance.
        self._policy = policy
        # Should jobs be packed.
        self._job_packing = 'Packing' in policy.name
        # RPC clients.
        self._cluster_spec = {}
        self._worker_connections = {}
        # Next job_id to assign.
        self._job_id_counter = 0
        # Next worker_id to assign.
        self._worker_id_counter = 0
        # Lock to ensure worker_id assignment is thread-safe.
        self._scheduler_lock = threading.Lock()
        # List of available worker IDs.
        self._available_worker_ids = queue.Queue()
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Iterations run on each worker_id, for all current incomplete
        # applications.
        self._steps_run_so_far = {}
        # Total number of iterations run for each incomplete job across
        # all worker types.
        self._total_steps_run = {}
        # Time run so far on each worker_id, for all current incomplete
        # applications.
        self._job_time_so_far = {}
        # Time spent running any application on each worker, for all current
        # incomplete applications.
        self._worker_time_so_far = {}
        # Cumulative time spent running any application on each worker.
        self._cumulative_worker_time_so_far = {}
        # Number of jobs to compute fair share.
        self._num_jobs = 0
        # Commands to run for all current incomplete applications.
        self._jobs = {}
        # Priority queues for each worker_type.
        self._priorities = {}
        self._deficits = {}
        # Number of failures per job.
        self._num_failures_per_job = {}
        # Timestamp when data structures recording elapsed time was last reset.
        self._last_reset_time = 0
        # Flag indicating when to update the allocation.
        self._need_to_update_allocation = False
        # Measured and predicted throughputs for all current incomplete
        # applications.
        self._throughputs = {}
        # Throughputs for all job types (pre-measured).
        if throughputs_file is not None:
            self._oracle_throughputs = utils.read_all_throughputs_json(
                throughputs_file)
        else:
            self._oracle_throughputs = None
        # Flag to indicate whether throughputs should be estimated online.
        self._estimate_throughputs = \
            self._job_packing and profiling_percentage > 0
        if self._estimate_throughputs:
            # Percentage of machines to use for profiling co-located jobs.
            self._profiling_percentage = profiling_percentage
            # Keeps track of which throughput values have been measured.
            self._throughputs_mask = {}
            # Initialize the throughputs matrix that newly arrived jobs will
            # be compared against.
            self._initialize_reference_throughputs(num_reference_models)
            # Keeps track of which jobs need to be profiled against
            # reference jobs.
            self._jobs_to_profile = {}
            # Keeps track of job throughputs when co-located with
            # reference jobs.
            self._profiled_jobs = {}
            # Map from job id to reference job type.
            self._reference_job_map = {}
        # Currently running jobs.
        self._running_jobs = set()
        # The timestamp when each worker entered the cluster.
        self._worker_start_times = {}
        # Verbose flag.
        self._verbose = False
        # Data structures for debugging.
        self._micro_tasks_per_job = {}
        self._all_jobs = []

        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'Done': self._done_callback,
        }

        if not self._simulate:
            self.server_thread = threading.Thread(
                target=scheduler_server.serve,
                args=(port, callbacks))
            self.server_thread.daemon = True
            self.server_thread.start()

            self.start_scheduling_thread()


    def _initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

        self._job_generator = random.Random()
        self._job_generator.seed(seed+2)

        self._interarrival_time_generator = random.Random()
        self._interarrival_time_generator.seed(seed+3)

        self._throughput_estimation_generator = np.random.RandomState()
        self._throughput_estimation_generator.seed(seed+4)

        self._worker_type_shuffler = random.Random()
        self._worker_type_shuffler.seed(seed+5)


    def start_scheduling_thread(self):
        self.scheduler_thread = threading.Thread(
            target=self.schedule,
            args=())
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()


    def _update_throughput(self, job_id, worker_type, all_num_steps,
                           all_execution_times):
        if self._simulate and self._estimate_throughputs:
            if not job_id.is_pair():
                # Assume single job throughputs are already populated.
                return
            elif (job_id.is_pair() and
                  not self._throughputs_mask[job_id][worker_type]):
                self._throughputs_mask[job_id][worker_type] = True
                oracle_throughputs = self._oracle_throughputs[worker_type]
                job_types = []
                for single_job_id in job_id.singletons():
                    job_types.append(self._jobs[single_job_id].job_type)
                self._throughputs[job_id][worker_type] = \
                    oracle_throughputs[job_types[0]][job_types[1]]
        elif not self._simulate:
            # Adjust the job throughput using an exponential moving average
            # between the old value and the new measurement.
            if job_id.is_pair():
                old_throughput = self._throughputs[job_id][worker_type]
            else:
                old_throughput = [self._throughputs[job_id][worker_type]]
            for i, single_job_id in enumerate(job_id.singletons()):
                new_throughput = all_num_steps[i] / all_execution_times[i]
                if old_throughput != INFINITY:
                    new_throughput *= EMA_ALPHA
                    new_throughput += (1 - EMA_ALPHA) * old_throughput[i]
                self._throughputs[job_id][worker_type][i] = new_throughput
            print(('[DEBUG] Job %s throughput on worker type %s: '
                   '%s -> %s') % (job_id, worker_type, str(old_throughput),
                                  str(self._throughputs[job_id][worker_type])))

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
            job: Job object to schedule. Contains information about the command
                 to run, as well as the number of steps to run the command for.
            timestamp (optional): Timestamp at which job is to be added
                                  (defaults to current_timestamp() if not
                                  specified).

        Returns:
            The job_id of the newly added job.
        """

        with self._scheduler_lock:
            current_timestamp = self.get_current_timestamp()
            job_id = job_id_pair.JobIdPair(self._job_id_counter, None)
            self._job_id_counter += 1
            job._job_id = job_id
            self._jobs[job_id] = job
            self._steps_run_so_far[job_id] = {}
            self._job_time_so_far[job_id] = {}
            self._throughputs[job_id] = {}
            self._num_failures_per_job[job_id] = 0
            self._total_steps_run[job_id] = 0
            for worker_type in self._worker_types:
                self._steps_run_so_far[job_id][worker_type] = 0
                self._set_initial_throughput(job_id, worker_type)
                if self._job_packing:
                    self._populate_job_combination_metadata(job_id,
                                                            worker_type)
                if self._estimate_throughputs:
                    self._jobs_to_profile[worker_type].add(job_id)

                self._job_time_so_far[job_id][worker_type] = \
                        (self._time_per_iteration / 2.0)
            self._per_job_start_timestamps[job_id] = current_timestamp
            self._per_job_latest_timestamps[job_id] = None
            self._add_to_priorities(job_id)
            self._need_to_update_allocation = True

            if timestamp is None:
                timestamp = self.get_current_timestamp()
            self._per_job_start_timestamps[job_id] = timestamp
            print('%s]\t[Job dispatched]\tJob ID: %s' % (timestamp, job_id))

        return job_id

    def remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        job_id = job_id_pair.JobIdPair(job_id, None)
        with self._scheduler_lock:
            duration = self._per_job_latest_timestamps[job_id] - \
                self._per_job_start_timestamps[job_id]
            self._job_completion_times[job_id] = duration
            self._job_priority_weights[job_id] = \
                self._jobs[job_id].priority_weight
            print("Job %d completed\n\tStart timestamp: %.2f\n\t"
                  "End timestamp: %.2f\nDuration: %.2f %s\n"
                  "Number of active jobs: %d\n" % (
                      job_id[0],
                      self._per_job_start_timestamps[job_id],
                      self._per_job_latest_timestamps[job_id],
                      duration, "seconds", len(self._jobs))
                  )

            del self._jobs[job_id]
            del self._steps_run_so_far[job_id]
            del self._total_steps_run[job_id]
            del self._job_time_so_far[job_id]
            del self._throughputs[job_id]
            del self._num_failures_per_job[job_id]
            if self._job_packing:
                to_delete = []
                for other_job_id in self._throughputs:
                    if (other_job_id.is_pair() and
                        job_id.overlaps_with(other_job_id)):
                        to_delete.append(other_job_id)
                for other_job_id in to_delete:
                    del self._throughputs[other_job_id]
                    del self._job_time_so_far[other_job_id]
                    if self._estimate_throughputs:
                        del self._throughputs_mask[other_job_id]

            self._remove_from_priorities(job_id)
            self._need_to_update_allocation = True

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
            return len(self._jobs) == 0

    def shutdown(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            for worker_id in self._worker_connections:
                self._worker_connections[worker_id].shutdown()
        # TODO: Any other cleanup?

    """
    ======================================================================
       Scheduler's main schedule() and simulate() methods.
    ======================================================================
    """

    def _profile_with_reference_jobs(self, worker_type, num_workers,
                                     num_profiling_jobs_per_machine=8):
        num_workers_left = num_workers
        num_profiling_machines = int(self._cluster_spec[worker_type] *
                                     self._profiling_percentage)
        num_profiling_jobs_per_machine = 8
        available_profiling_slots = \
            num_profiling_machines * num_profiling_jobs_per_machine
        used_profiling_slots = 0
        job_types_to_profile = {}
        oracle_throughputs = self._oracle_throughputs[worker_type]

        # Randomly select an order of reference jobs for each newly arrived
        # job to co-locate with.
        jobs_to_profile = list(self._jobs_to_profile[worker_type])
        for job_id in jobs_to_profile:
            self._profiled_jobs[worker_type][job_id] = {}
            job_types_to_profile[job_id] = \
                self._throughput_estimation_generator.choice(
                        self._reference_job_types,
                        len(self._reference_job_types),
                        replace=False).tolist()
        self._jobs_to_profile[worker_type] = set()

        # Fill in all profiling slots by round-robining between each job.
        while (used_profiling_slots < available_profiling_slots and
               len(job_types_to_profile) > 0):
            completed_jobs = []
            for job_id in sorted(job_types_to_profile.keys()):
                job_type = self._jobs[job_id].job_type
                reference_job_type = job_types_to_profile[job_id].pop(0)
                self._profiled_jobs[worker_type][job_id][reference_job_type] = \
                    oracle_throughputs[job_type][reference_job_type] 
                    #(oracle_throughputs[job_type][reference_job_type][0] / \
                    # oracle_throughputs[job_type]['null'])
                used_profiling_slots += 1
                if len(job_types_to_profile[job_id]) == 0:
                    completed_jobs.append(job_id)
            for completed_job_id in completed_jobs:
                del job_types_to_profile[completed_job_id]

        num_workers_left -= int(math.ceil(used_profiling_slots / \
                                          num_profiling_jobs_per_machine))
        return num_workers_left


    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper(self, worker_type,
                                         already_scheduled_jobs):
        """Greedily selects the jobs to run in the next round by iterating
           through the job list in sorted priority order.

           Assumes only single-GPU jobs.

           Returns:
             A list of job IDs to schedule on the passed-in worker_type in
             the upcoming round.
        """
        num_workers = len(
            self._worker_type_to_worker_id_mapping[worker_type])
        already_scheduled_jobs_set = set(already_scheduled_jobs)
        scheduled_jobs_on_worker_type = []

        if self._estimate_throughputs:
            num_workers_left = self._profile_with_reference_jobs(worker_type,
                                                                 num_workers)
        else:
            num_workers_left = num_workers

        entries = []
        for job_id in self._priorities[worker_type]:
            entries.append((job_id, self._priorities[worker_type][job_id],
                            self._deficits[worker_type][job_id],
                            self._allocation[job_id][worker_type]))

        sorted_job_queue = sorted(entries,
                                  key=lambda x: (x[1], x[2], x[3]),
                                  reverse=True)

        for job_id, *_ in sorted_job_queue:
            if num_workers_left == 0:
                break
            # Don't schedule jobs that have already been scheduled.
            if ((not job_id.is_pair() and job_id in already_scheduled_jobs_set) or
                (job_id.is_pair() and
                 (job_id.singletons()[0] in already_scheduled_jobs_set or
                  job_id.singletons()[1] in already_scheduled_jobs_set))):
                continue

            # Don't schedule jobs with 0 throughput.
            if ((job_id.is_pair() and
                (self._throughputs[job_id][worker_type][0] <= 0 or
                 self._throughputs[job_id][worker_type][1] <= 0)) or
                (not job_id.is_pair() and
                 self._throughputs[job_id][worker_type] <= 0)):
                continue

            # For FIFO jobs, don't schedule jobs with 0 priority.
            if (self._policy.name.startswith("FIFO") and
                self._priorities[worker_type][job_id] <= 0.0):
                continue

            # Make sure job fits in remaining number of workers.
            # If not, move onto next job.
            if job_id.is_pair():
                scale_factor = \
                    self._jobs[job_id.singletons()[0]].scale_factor
                other_scale_factor = \
                    self._jobs[job_id.singletons()[1]].scale_factor
                # Only pack jobs with the same scale_factor.
                if scale_factor != other_scale_factor:
                    continue
            else:
                scale_factor = self._jobs[job_id].scale_factor
            if scale_factor > num_workers_left:
                continue
            num_workers_left -= scale_factor

            for single_job_id in job_id.singletons():
                already_scheduled_jobs_set.add(single_job_id)
            scheduled_jobs_on_worker_type.append((job_id,
                                                  scale_factor))

        return scheduled_jobs_on_worker_type


    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers(self):
        """Attempts to schedule jobs on as many alive workers as possible.

           Returns:
             A list of job IDs and tuple of worker IDs for each scheduled job
             in the coming round.
        """

        # Update priorities before trying to figure out applications to run
        # in the upcoming round.
        self._update_priorities()

        already_scheduled_jobs = []
        scheduled_jobs = []

        to_remove = []
        worker_types = ["v100", "p100", "k80"]

        # TODO: Replace this with a single global queue containing all worker types.
        if "Perf" not in self._policy.name and "Packing" not in self._policy.name:
            self._worker_type_shuffler.shuffle(worker_types)

        for i, worker_type in enumerate(worker_types):
            if worker_type not in self._worker_type_to_worker_id_mapping:
                to_remove.append(i)
        for i in reversed(to_remove):
            worker_types.pop(i)

        for worker_type in worker_types:
            worker_ids = self._worker_type_to_worker_id_mapping[worker_type]
            worker_id_ptr = 0
            scheduled_jobs_on_worker_type = \
                    self._schedule_jobs_on_workers_helper(
                            worker_type, already_scheduled_jobs)

            for (job_id, scale_factor) in scheduled_jobs_on_worker_type:
                # Make sure a job is only scheduled on a single worker_type in
                # a given round.
                already_scheduled_jobs.append(job_id)
                if job_id.is_pair():
                    for single_job_id in job_id.singletons():
                        already_scheduled_jobs.append(single_job_id)

                # For now, ignore locality. Place job_id on the first
                # `scale_factor` workers of the desired type.
                # assert(scale_factor == self._jobs[job_id].scale_factor)
                worker_id_ptrs = [worker_id_ptr + i for i in range(scale_factor)]
                scheduled_jobs.append((job_id,
                                       tuple([worker_ids[i] for i in worker_id_ptrs])))
                worker_id_ptr += scale_factor

                for single_job_id in job_id.singletons():
                    num_steps = self._get_num_steps(job_id, worker_type,
                                                    single_job_id)
                    if not self._estimate_throughputs and num_steps <= 0:
                        raise ValueError('Num steps should be greater '
                                         'than 0, is %d (Job ID: %s, '
                                         'job_type=%s, '
                                         'worker_type=%s)' % (num_steps,
                                                              job_id,
                                                              self._jobs[job_id].job_type,
                                                              worker_type))
                    self._per_job_latest_timestamps[single_job_id] = \
                        self.get_current_timestamp()
                    self._running_jobs.add(single_job_id)
                worker_types = []
                for x in self._allocation[job_id]:
                    worker_types.append(x)
                worker_types = sorted(worker_types)
                allocation_str = ''
                for x in worker_types:
                    allocation_str += ' [%4s %f]' % (x, self._allocation[job_id][x])
                print(('%s]\t[Micro-task scheduled]\tJob ID: %s\t'
                       'Worker type: %s\tWorker ID(s): %s\t'
                       'Priority: %f\tDeficit: %f\t'
                       'Allocation: %s') % (self.get_current_timestamp(),
                                           job_id, worker_type,
                                           ",".join(["%d" % worker_ids[i]
                                                     for i in worker_id_ptrs]),
                                           self._priorities[worker_type][job_id],
                                           self._deficits[worker_type][job_id],
                                           allocation_str))
            if worker_id_ptr < len(worker_ids):
                print(('WARNING: %d GPUs of type %s left unused. '
                       'Number of active jobs: %d') % (len(worker_ids) - worker_id_ptr,
                                                       worker_type,
                                                       len(self._jobs)))

        return scheduled_jobs

    def _get_num_steps(self, job_id, worker_type, single_job_id=None):
        if job_id.is_pair():
            assert(single_job_id is not None)
            index = job_id.as_tuple().index(single_job_id[0])
            num_steps = int(self._throughputs[job_id][worker_type][index] *
                            self._time_per_iteration)
        else:
            num_steps = int(self._throughputs[job_id][worker_type] *
                            self._time_per_iteration)
        return min(num_steps,
                   self._get_remaining_steps(single_job_id))

    def _get_job_steps_and_finish_times(self, job_id, worker_type):
        """Returns the number of steps to execute and and latest finish time(s)
           for a job or job pair."""
        max_finish_time = self.get_current_timestamp()
        all_num_steps = []
        single_job_ids = job_id.singletons()
        if job_id.is_pair() and self._estimate_throughputs and self._simulate:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            job_types = []
            for single_job_id in single_job_ids:
                job_types.append(self._jobs[single_job_id].job_type)
            oracle_throughput = oracle_throughputs[job_types[0]][job_types[1]]
        for i, single_job_id in enumerate(single_job_ids):
            num_steps = self._get_num_steps(job_id, worker_type, single_job_id)
            all_num_steps.append(num_steps)
            if job_id.is_pair():
                if self._estimate_throughputs and self._simulate:
                    throughput = oracle_throughput[i]
                else:
                    throughput = self._throughputs[job_id][worker_type][i]
            else:
                # NOTE: Assumes single job throughputs are accurate in
                # simulation + estimation case.
                throughput = self._throughputs[job_id][worker_type]
            if throughput <= 0:
                if self._estimate_throughputs:
                    all_num_steps.append(0)
                    finish_time = max_finish_time
                else:
                    print(single_job_id)
                    print(worker_type)
                    raise Exception("Throughput should not be less than 0!")
            else:
                execution_time = num_steps / throughput
                finish_time = (self.get_current_timestamp() + \
                                (num_steps / throughput))
            if finish_time > max_finish_time:
                max_finish_time = finish_time
            self._running_jobs.add(single_job_id)
        return all_num_steps, max_finish_time


    def _save_checkpoint(self, checkpoint_file, completed_jobs,
                         last_job_arrival_time,
                         next_job_arrival_time,
                         current_round_start_time,
                         current_round_end_time,
                         running_jobs):
        with open(checkpoint_file, 'wb') as f:
            import pickle
            pickle.dump(completed_jobs, f)
            pickle.dump(last_job_arrival_time, f)
            pickle.dump(next_job_arrival_time, f)
            pickle.dump(current_round_start_time, f)
            pickle.dump(current_round_end_time, f)
            pickle.dump(running_jobs, f)

            pickle.dump(self._jobs, f)
            pickle.dump(self._throughputs, f)
            if self._estimate_throughputs:
                pickle.dump(self._throughputs_mask, f)
                pickle.dump(self._jobs_to_profile, f)
                pickle.dump(self._profiled_jobs, f)
                pickle.dump(self._reference_job_map, f)
            pickle.dump(self._allocation, f)
            pickle.dump(self._steps_run_so_far, f)
            pickle.dump(self._total_steps_run, f)
            pickle.dump(self._job_time_so_far, f)
            pickle.dump(self._worker_start_times, f)
            pickle.dump(self._worker_time_so_far, f)
            pickle.dump(self._cumulative_worker_time_so_far, f)
            pickle.dump(self._num_jobs, f)
            pickle.dump(self._priorities, f)
            pickle.dump(self._deficits, f)
            pickle.dump(self._last_reset_time, f)
            pickle.dump(self._need_to_update_allocation, f)
            pickle.dump(self._job_generator, f)
            pickle.dump(self._interarrival_time_generator, f)
            pickle.dump(self._per_job_start_timestamps, f)
            pickle.dump(self._per_job_latest_timestamps, f)
            pickle.dump(self._job_completion_times, f)
            pickle.dump(self._current_timestamp, f)
            pickle.dump(self._job_id_counter, f)


    def _load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            import pickle
            completed_jobs = pickle.load(f)
            last_job_arrival_time = pickle.load(f)
            next_job_arrival_time = pickle.load(f)
            current_round_start_time = pickle.load(f)
            current_round_end_time = pickle.load(f)
            running_jobs = pickle.load(f)

            self._jobs = pickle.load(f)
            self._throughputs = pickle.load(f)
            if self._estimate_throughputs:
                self._throughputs_mask = pickle.load(f)
                self._jobs_to_profile = pickle.load(f)
                self._profiled_jobs = pickle.load(f)
                self._reference_job_map = pickle.load(f)
                self._profiled_job_combinations = pickle.load(f)
            self._allocation = pickle.load(f)
            self._steps_run_so_far = pickle.load(f)
            self._total_steps_run = pickle.load(f)
            self._job_time_so_far = pickle.load(f)
            self._worker_start_times = pickle.load(f)
            self._worker_time_so_far = pickle.load(f)
            self._cumulative_worker_time_so_far = pickle.load(f)
            self._num_jobs = pickle.load(f)
            self._priorities = pickle.load(f)
            self._deficits = pickle.load(f)
            self._last_reset_time = pickle.load(f)
            self._need_to_update_allocation = pickle.load(f)
            self._job_generator = pickle.load(f)
            self._interarrival_time_generator = pickle.load(f)
            self._per_job_start_timestamps = pickle.load(f)
            self._per_job_latest_timestamps = pickle.load(f)
            self._job_completion_times = pickle.load(f)
            self._current_timestamp = pickle.load(f)
            self._job_id_counter = pickle.load(f)

            return (completed_jobs,
                    last_job_arrival_time,
                    next_job_arrival_time,
                    current_round_start_time,
                    current_round_end_time,
                    running_jobs)


    def _sample_arrival_time_delta(self, rate_parameter):
        """Samples job interarrival rate from a Poisson distribution according
           to the specified rate parameter."""
        return -math.log(1.0 - self._interarrival_time_generator.random()) / rate_parameter

    def _generate_job(self, fixed_job_duration=None,
                      generate_multi_gpu_jobs=False,
                      generate_multi_priority_jobs=False,
                      run_dir='/tmp'):
        """Generates a new job for simulation."""
        job_template = self._job_generator.choice(JobTable)
        job_type = job_template.model
        if fixed_job_duration:
            print('Running for fixed duration %d minutes' % (fixed_job_duration / 60.0))
            run_time = fixed_job_duration
        else:
            run_time = 60 * (10 ** self._job_generator.uniform(2, 4))
        num_steps = \
            run_time * self._oracle_throughputs['v100'][job_type]['null']
        assert(run_time > 0)
        assert(num_steps > 0)
        if job_template.needs_data_dir:
            command = job_template.command % (run_dir, run_dir)
        else:
            command = job_template.command % (run_dir)

        scale_factor = 1
        if generate_multi_gpu_jobs:  # Copies Philly distribution.
            r = self._job_generator.uniform(0, 1)
            if 0.8 <= r <= 0.85:
                scale_factor = 2
            elif 0.85 <= r <= 0.95:
                scale_factor = 4
            elif 0.95 <= r:
                scale_factor = 8

        priority_weight = 1.0
        if generate_multi_priority_jobs:
            r = self._job_generator.uniform(0, 1)
            if 0.0 <= r <= 0.2:
                priority_weight = 5.0

        job = Job(job_id=None,
                  job_type=job_type,
                  command=command,
                  num_steps_arg=job_template.num_steps_arg,
                  total_steps=num_steps,
                  duration=None,
                  scale_factor=scale_factor,
                  priority_weight=priority_weight)

        return job

    def simulate(self, cluster_spec, arrival_times=None, jobs=None,
                 measure_steady_state_jobs=False, lam=None,
                 jobs_to_complete=None,
                 fixed_job_duration=None, num_total_jobs=None,
                 generate_multi_gpu_jobs=False,
                 generate_multi_priority_jobs=False,
                 simulate_steady_state=False, debug=False,
                 checkpoint_threshold=None,
                 checkpoint_file=None):
        """Simulates the scheduler execution.

           Simulation can be performed using a trace or with continuously
           generated synthetic data. Simulation is terminated when either
               1) All jobs in the specified trace complete.
               2) A specific subset of jobs complete.
               3) All jobs in a specific time window complete.

           Currently, the cluster specification must be statically
           specified from the beginning of execution.

           Args:
            cluster_spec: A dictionary of worker type to worker count.
            arrival_times: The arrival times of a set of pre-generated jobs.
            jobs: A set of pre-generated jobs.
            lam: 1 / the rate parameter to be passed in to the Poisson process
                 used to generate arrival times.
            jobs_to_complete: A set of `JobIdPair`s that must be completed
                              before terminating the simulation.
            fixed_job_duration: If set, all generated jobs will have this
                                duration if run exclusively on a v100.
            num_total_jobs: If set, only `num_total_jobs` jobs will
                            be generated.
            generate_multi_gpu_jobs: If set, some jobs will have `scale_factor`
                                     greater than 1, according to a pre-defined
                                     distribution.
            generate_multi_priority_jobs: If set, 20% of jobs will have a
                                          priority of 5.0.
            simulate_steady_state: If set, adds as many jobs as there are
                                   workers before beginning the simulation.
            debug: If set, pauses the simulation at the start of every loop.
        """

        from_trace = arrival_times is not None and jobs is not None
        if num_total_jobs is not None:
            remaining_jobs = num_total_jobs
        if from_trace:
            remaining_jobs = len(jobs)
            queued_jobs = []
        else:
            if self._oracle_throughputs is None:
                raise ValueError('Scheduler must be initialized with a '
                                 'throughputs file.')
            elif lam is None:
                raise ValueError('\'lam\' must be specified when running '
                                 'without trace.')
        if (not from_trace and jobs_to_complete is None and
            num_total_jobs is None):
            raise ValueError('One of \'jobs_to_complete\' '
                             'or \'num_total_jobs\' must be set.')
        if (checkpoint_file is not None and (from_trace or simulate_steady_state)):
            raise ValueError('Checkpointing only intended to be used '
                             'when generating trace on-the-fly.')

        running_jobs = []
        num_jobs_generated = 0
        completed_jobs = set()
        last_job_arrival_time = None
        next_job_arrival_time = 0
        no_dispatched_or_running_jobs = False
        current_round_start_time = 0
        current_round_end_time = None
        num_completed_jobs = 0

        # Set up the cluster according to the provided spec.
        worker_types = sorted([worker_type for worker_type in cluster_spec])
        for worker_type in worker_types:
            for i in range(cluster_spec[worker_type]):
                self._register_worker_callback(worker_type)

        if checkpoint_file is not None and checkpoint_threshold is None:
            (completed_jobs,
             last_job_arrival_time,
             next_job_arrival_time,
             current_round_start_time,
             current_round_end_time,
             running_jobs) = self._load_checkpoint(checkpoint_file)

        if from_trace:
            # Add all jobs to the queue.
            for i in range(1, len(arrival_times)):
                assert(arrival_times[i] >= arrival_times[i-1])

            for (arrival_time, job) in zip(arrival_times, jobs):
                queued_jobs.append((arrival_time, job))
        elif simulate_steady_state:
            for worker_type in worker_types:
                for i in range(cluster_spec[worker_type]):
                    job = self._generate_job(
                        fixed_job_duration=fixed_job_duration,
                        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                        generate_multi_priority_jobs=generate_multi_priority_jobs)
                    num_jobs_generated += 1
                    self._all_jobs.append((0, job))
                    job_id = self.add_job(job, timestamp=0)

        while True:
            if debug:
                input('Press Enter to continue...')
            if (jobs_to_complete is not None and
                  jobs_to_complete.issubset(completed_jobs)):
                break
            elif (num_total_jobs is not None and
                    remaining_jobs <= 0):
                break
            elif from_trace:
                if remaining_jobs == 0:
                    break
                elif len(queued_jobs) > 0:
                    next_job_arrival_time = queued_jobs[0][0]
                else:
                    next_job_arrival_time = None

            # Jump to the next event's timestamp.
            # Find the time when the latest job completes, which signals
            # the finishing of the round.
            max_timestamp = 0
            if (len(running_jobs) > 0 and
                -running_jobs[0][0] > max_timestamp):
                max_timestamp = -running_jobs[0][0]
                if current_round_end_time is not None:
                    current_round_start_time = current_round_end_time
                current_round_end_time = max_timestamp
            if max_timestamp > 0:
                self._current_timestamp = max_timestamp
            else:
                self._current_timestamp = next_job_arrival_time

            # Check if any jobs have completed.
            while len(running_jobs) > 0:
                (finish_time, job_id, worker_ids, all_num_steps) = \
                        running_jobs[0]
                finish_time = (-finish_time)
                if finish_time <= self._current_timestamp:
                    all_execution_times = []
                    for single_job_id in job_id.singletons():
                        start_time = current_round_start_time
                        execution_time = finish_time - start_time
                        all_execution_times.append(execution_time)
                        self._per_job_latest_timestamps[single_job_id] = \
                                finish_time
                    # TODO: decide whether to pass in all worker_ids to
                    # _done_callback.
                    for worker_id in worker_ids:
                        self._done_callback(job_id, worker_id,
                                            all_num_steps,
                                            all_execution_times)
                    for single_job_id in job_id.singletons():
                        if single_job_id not in self._jobs:
                            completed_jobs.add(single_job_id)
                            if from_trace or num_total_jobs is not None:
                                remaining_jobs -= 1
                    heapq.heappop(running_jobs)
                else:
                    break

            # Since we're scheduling in rounds, no jobs should be
            # running when scheduling the next round of jobs.
            assert(len(running_jobs) == 0)

            # Dispatch any newly arrived jobs.
            last_added_job_id = None
            if from_trace:
                while len(queued_jobs) > 0:
                    (arrival_time, job) = queued_jobs[0]
                    if arrival_time <= self._current_timestamp:
                        job_id = self.add_job(job, timestamp=arrival_time)
                        last_added_job_id = job_id
                        queued_jobs.pop(0)
                    else:
                        break
            else:
                while next_job_arrival_time <= self._current_timestamp:
                    if num_total_jobs is not None:
                        if num_jobs_generated > num_total_jobs:
                            break
                    job = self._generate_job(
                        fixed_job_duration=fixed_job_duration,
                        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                        generate_multi_priority_jobs=generate_multi_priority_jobs)
                    num_jobs_generated += 1
                    self._all_jobs.append((next_job_arrival_time, job))
                    job_id = self.add_job(job, timestamp=next_job_arrival_time)
                    last_added_job_id = job_id

                    last_job_arrival_time = next_job_arrival_time
                    if lam == 0.0:
                        arrival_time_delta = 0.0
                    else:
                        arrival_time_delta = \
                                self._sample_arrival_time_delta(1.0 / lam)
                    next_job_arrival_time = \
                            arrival_time_delta + last_job_arrival_time

            # Schedule jobs until there are no available workers or no jobs
            # with non-zero allocations on available workers.
            scheduled_jobs = self._schedule_jobs_on_workers()
            for (job_id, worker_ids) in scheduled_jobs:
                worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
                for worker_id in worker_ids:
                    self._remove_available_worker_id(worker_id)
                all_num_steps, max_finish_time = \
                        self._get_job_steps_and_finish_times(job_id,
                                                             worker_type)
                heapq.heappush(running_jobs, (-max_finish_time, job_id,
                                              worker_ids,
                                              all_num_steps))

            if checkpoint_threshold is not None and last_added_job_id is not None \
                and last_added_job_id[0] >= checkpoint_threshold \
                and not checkpoint_complete:
                # Create checkpoint.
                assert(checkpoint_file is not None)
                self._save_checkpoint(checkpoint_file,
                                      completed_jobs,
                                      last_job_arrival_time,
                                      next_job_arrival_time,
                                      current_round_start_time,
                                      current_round_end_time,
                                      running_jobs)
                checkpoint_complete = True

        print('Total duration: %.3f seconds' % (self._current_timestamp))

    def _schedule_with_rounds(self):
        """Schedules jobs on workers using rounds.

        In a loop, schedules in rounds the applications most in need of
        being run (that is, the applications with the highest
        fraction_allocated/fraction_run ratio) using a DP algorithm.
        """

        while True:
            with self._scheduler_lock:
                num_workers = len(self.worker_ids)
                # Reset available_worker_ids to the desired size.
                self._available_worker_ids = queue.Queue(self.num_workers)
                for worker_id in self.worker_ids:
                    self._add_available_worker_id(worker_id)
                scheduled_jobs = self._schedule_jobs_on_workers()
                for (job_id, worker_ids) in scheduled_jobs:
                    worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
                    # TODO: Support packing.
                    num_steps = self._get_num_steps(job_id, worker_type)
                    for worker_id in worker_ids:
                        self._worker_connections[worker_id].run(
                            [(job_id[0], self._jobs[job_id].command,
                              self._jobs[job_id].num_steps_arg,
                              num_steps)])
                        self._remove_available_worker_id(worker_id)
            self._wait_until_all_workers_available(num_workers)

    def schedule(self):
        """Schedules jobs on workers."""
        self._schedule_with_rounds()


    def get_average_jct(self, job_ids=None):
        """Computes the average job completion time.

           Args:
               job_ids: A list of `JobIdPair` objects. If specified, computes
                        the average JCT using only these jobs.

           Returns: The average JCT.
        """
        with self._scheduler_lock:
            if len(self._job_completion_times) == 0:
                return
            if job_ids is None:
                job_ids = sorted([job_id for job_id in self._job_completion_times])
            print('Job completion times:')
            low_priority_job_completion_times = []
            high_priority_job_completion_times = []
            for job_id in job_ids:
                if self._job_priority_weights[job_id] == 1.0:
                    print('Job %s: %.3f' % (job_id,
                                            self._job_completion_times[job_id]))
                    low_priority_job_completion_times.append(
                        self._job_completion_times[job_id])
                else:
                    print('Job %s (high priority): %.3f' % (job_id,
                                                            self._job_completion_times[job_id]))
                    high_priority_job_completion_times.append(
                        self._job_completion_times[job_id])
            average_job_completion_time = \
                np.mean([self._job_completion_times[job_id] for job_id in job_ids])
            print('Average job completion time: '
                  '%.3f seconds' % (average_job_completion_time))
            if len(low_priority_job_completion_times) > 0:
                print('Average job completion time (low priority): '
                      '%.3f seconds' % (np.mean(low_priority_job_completion_times)))
            if len(high_priority_job_completion_times) > 0:
                print('Average job completion time (high priority): '
                      '%.3f seconds' % (np.mean(high_priority_job_completion_times)))
            return average_job_completion_time


    def get_cluster_utilization(self):
        """Computes the utilization of the cluster."""
        with self._scheduler_lock:
            utilizations = []
            current_timestamp = self.get_current_timestamp()
            for worker_id in self._cumulative_worker_time_so_far:
                total_runtime = (current_timestamp -
                                 self._worker_start_times[worker_id])
                worker_time = self._cumulative_worker_time_so_far[worker_id]
                utilization = worker_time / total_runtime
                if utilization > 1.0 and not self._job_packing:
                    print('Error: invalid utilization %.3f' % (utilization))
                    print('Worker ID: %d' % (worker_id))
                    print('Worker time: %.3f' % (worker_time))
                    print('Total time: %.3f.' % (total_runtime))
                    return None
                utilizations.append(utilization)
            cluster_utilization = np.mean(utilizations)
            print('Cluster utilization: %.3f' % (cluster_utilization))
            return cluster_utilization

    def get_micro_tasks(self):
        """Prints all micro-tasks run for each job.

           Debug function used print all micro-tasks run for each job.
        """
        job_ids = sorted(self._micro_tasks_per_job.keys())
        for job_id in job_ids:
            print('Job %s: %d' % (job_id, len(self._micro_tasks_per_job[job_id])))
            for i, (start, end) in enumerate(self._micro_tasks_per_job[job_id]):
                print('\t%d%f - %f' % (i, start, end))
            print('')

    def get_job_start_and_end_times(self):
        """Returns the start and end times of each job.

           Debug function for returning the start and end times of each job.
        """
        with self._scheduler_lock:
            job_ids = sorted(
                [job_id for job_id in self._per_job_latest_timestamps])
            start_times = [
                self._per_job_start_timestamps[job_id]
                for job_id in job_ids]
            end_times = [
                self._per_job_latest_timestamps[job_id]
                for job_id in job_ids]
        return start_times, end_times

    def get_all_simulated_jobs(self, job_range):
        """Returns all the jobs run during simulation.

           Debug function used to print all jobs generated during
           simulation within a specified range.

           Args:
               job_range: A tuple specifying which jobs to be printed.
        """
        print('All simulated jobs:')
        for arrival_time, job in self._all_jobs[job_range[0]:job_range[1]]:
            print('%s\t%s\t%d\t%f' % (job.job_id,
                                      job.job_type,
                                      job.total_steps,
                                      arrival_time))
    """
    ======================================================================
       Helper methods to get and mutate state needed for scheduling.
    ======================================================================
    """

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _print_allocation(self):
        """Prints the allocation.

           Debug method used for printing the allocation of each job on each
           worker type.
        """
        print('')
        print('=' * 80)
        print('Allocation\t(Current_time: %f)' % (self.get_current_timestamp()))
        print('-' * 80)
        for job_id in sorted(list(self._allocation.keys())):
            allocation_str = 'Job ID %s:' % (job_id)
            for worker_type in sorted(list(self._allocation[job_id].keys())):
                allocation = self._allocation[job_id][worker_type]
                allocation_str += ' [%s: %f]' % (worker_type, allocation)
            print(allocation_str)
        print('=' * 80)
        print('')

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _print_deficits(self):
        """Prints the deficit.

           Debug method used for printing the deficit of each job on each
           worker type.
        """
        print('')
        print('=' * 80)
        print('Deficits\t(Current_time: %f)' % (self.get_current_timestamp()))
        print('-' * 80)
        for job_id in sorted(list(self._jobs.keys())):
            deficit_str = 'Job ID %s:' % (job_id)
            for worker_type in sorted(self._worker_types):
                deficit = self._deficits[worker_type][job_id]
                deficit_str += ' [%s: %f]' % (worker_type, deficit)
            print(deficit_str)
        print('=' * 80)
        print('')

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
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

        scale_factors = {
            job_id: self._jobs[job_id].scale_factor
            for job_id in self._jobs
        }
        if self._policy.name.startswith("MaxMinFairness"):
            priority_weights = {
                job_id: self._jobs[job_id].priority_weight
                for job_id in self._jobs
            }
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, priority_weights,
                self._cluster_spec)
        elif self._policy.name.startswith("MinTotalDuration"):
            num_steps_remaining = {
                job_id: self._get_remaining_steps(job_id)
                for job_id in self._jobs}
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, num_steps_remaining,
                self._cluster_spec)
        else:
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, self._cluster_spec)
        if unflattened_allocation is None:
            return None

        return unflattened_allocation

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _populate_job_combination_metadata(self, job_id, worker_type):
        """Populate metadata for job combinations involving passed-in job_id."""

        job = self._jobs[job_id]
        for other_job_id in self._jobs:
            if other_job_id != job_id:
                other_job = self._jobs[other_job_id]
                merged_job_id = \
                        job_id_pair.JobIdPair(job_id[0], other_job_id[0])
                if merged_job_id not in self._throughputs:
                    self._throughputs[merged_job_id] = {}
                    if self._estimate_throughputs:
                        self._throughputs_mask[merged_job_id] = {}
                    self._job_time_so_far[merged_job_id] = {}
                    self._priorities[worker_type][job_id] = 0.0
                    self._deficits[worker_type][job_id] = 0.0
                if (job.scale_factor != other_job.scale_factor or
                    self._estimate_throughputs):
                    self._throughputs[merged_job_id][worker_type] = [0.0, 0.0]
                    if self._estimate_throughputs:
                        self._throughputs_mask[merged_job_id][worker_type] = False
                else:
                    oracle_throughputs = self._oracle_throughputs[worker_type]
                    # The single-job IDs for job pairs are stored in sorted
                    # order so make sure the co-located throughputs match this
                    # order.
                    if job_id < other_job_id:
                        self._throughputs[merged_job_id][worker_type] = \
                            oracle_throughputs[job.job_type][other_job.job_type]
                    else:
                        self._throughputs[merged_job_id][worker_type] = \
                            oracle_throughputs[other_job.job_type][job.job_type]

    def _set_initial_throughput(self, job_id, worker_type):
        assert(not job_id.is_pair())
        if self._simulate:
            job_type = self._jobs[job_id].job_type
            self._throughputs[job_id][worker_type] = \
                self._oracle_throughputs[worker_type][job_type]['null']
        else:
            self._throughputs[job_id][worker_type] = DEFAULT_THROUGHPUT

    def _initialize_reference_throughputs(self, num_reference_models):
        self._reference_throughputs = {}
        all_worker_types = sorted(self._oracle_throughputs.keys())
        all_job_types = \
            sorted(self._oracle_throughputs[all_worker_types[0]].keys())
        self._reference_job_types = \
            self._throughput_estimation_generator.choice(
                    all_job_types, num_reference_models, replace=False)
        for worker_type in self._oracle_throughputs:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            self._reference_throughputs[worker_type] = \
                np.zeros((num_reference_models, num_reference_models),
                         dtype=np.float32)
            for i, job_type_0 in enumerate(self._reference_job_types):
                for j, job_type_1 in enumerate(self._reference_job_types):
                    if j < i:
                        continue
                    isolated_throughputs = []
                    for job_type in [job_type_0, job_type_1]:
                        isolated_throughputs.append(
                            oracle_throughputs[job_type]['null'])
                    self._reference_throughputs[worker_type][i][j] = \
                        oracle_throughputs[job_type_0][job_type_1][0] 
                        #(oracle_throughputs[job_type_0][job_type_1][0] /
                        #    isolated_throughputs[0])
                    self._reference_throughputs[worker_type][j][i] = \
                       oracle_throughputs[job_type_0][job_type_1][1]
                       #(oracle_throughputs[job_type_0][job_type_1][1] /
                       #     isolated_throughputs[1])
        noise = \
            self._throughput_estimation_generator.normal(0, 0.05,
                                                        (num_reference_models,
                                                         num_reference_models))
        self._reference_throughputs[worker_type] += noise
        np.clip(self._reference_throughputs[worker_type], 0, 1,
                out=self._reference_throughputs[worker_type])

    def _cosine_distance(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _match_job_to_reference_job(self, job_id, worker_type):
        num_reference_job_types = len(self._reference_job_types)
        oracle_throughputs = self._oracle_throughputs[worker_type]

        # Add a row to reference throughputs for the newly arrived job.
        throughputs_matrix = np.zeros((num_reference_job_types+1,
                                       num_reference_job_types+1),
                                      dtype=np.float32)
        throughputs_matrix[:-1,:-1]= \
            self._reference_throughputs[worker_type]
        """
        throughputs_matrix = \
            np.concatenate((self._reference_throughputs[worker_type],
                            np.zeros((1, num_reference_job_types),
                                     dtype=np.float32)),
                           axis=0)
        """
        
        # Initialize the mask.
        mask = np.zeros((num_reference_job_types+1, num_reference_job_types+1),
                         dtype=np.float32)
        mask[:num_reference_job_types][:num_reference_job_types] = 1.0
        
        """
        mask = np.concatenate((np.ones((num_reference_job_types,
                                        num_reference_job_types),
                                       dtype=np.float32),
                               np.zeros((1, num_reference_job_types),
                                        dtype=np.float32)),
                              axis=0)
        """

        # Fill in measured data points.
        for i, reference_job_type in enumerate(self._reference_job_types):
            if reference_job_type in self._profiled_jobs[worker_type][job_id]:
                throughputs_matrix[-1][i] = \
                    self._profiled_jobs[worker_type][job_id][reference_job_type][0]
                throughputs_matrix[i][-1] = \
                    self._profiled_jobs[worker_type][job_id][reference_job_type][1]
                mask[-1][i] = 1
                mask[i][-1] = 1


        # Run matrix completion algorithm if there are values to estimate.
        if (np.min(mask) == 0):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                k = DEFAULT_MATRIX_COMPLETION_K
                mu = DEFAULT_MATRIX_COMPLETION_MU
                try:
                    estimated_throughputs = \
                        matrix_completion.pmf_solve(throughputs_matrix,
                                                            mask,
                                                            k=k,
                                                            mu=mu)#,
                                #0.0, 1.0)
                    for i in range(len(mask)):
                        for j in range(len(mask[0])):
                            if not mask[i][j]:
                                throughputs_matrix[i][j] = \
                                    estimated_throughputs[i][j]
                except np.linalg.LinAlgError as e:
                    print('WARNING: could not estimate throughputs!')
                    print(e)
                    estimated_throughputs = None

        # Normalize the estimated throughputs.
        #throughputs_matrix /= np.linalg.norm(throughputs_matrix, ord=2, axis=1,
        #                                     keepdims=True)

        #print('throughputs_matrix:')
        #print(throughputs_matrix)

        # Measure the distance from the new row to every other row and find
        # the row with the smallest distance.
        distances = []
        for i, reference_job_type in enumerate(self._reference_job_types):
            distances.append((reference_job_type,
                #self._cosine_distance(throughputs_matrix[i][:-1], throughputs_matrix[-1][:-1])))
                self._euclidean_distance(throughputs_matrix[i][:-1], throughputs_matrix[-1][:-1])))
                                
            #np.linalg.norm(throughputs_matrix[i][:-1] -
            #                                 throughputs_matrix[-1][:-1])))
        distances.sort(key=lambda x: x[1])
        predicted_job_type = distances[0][0]
        self._reference_job_map[job_id] = predicted_job_type

        if predicted_job_type == self._jobs[job_id].job_type:
            self._num_correct_predictions += 1
 
        print('Job %s: Original job type=%s, '
              'Predicted job type=%s '
              '(%d correct predictions)' % (job_id,
                                            self._jobs[job_id].job_type,
                                            predicted_job_type,
                                            self._num_correct_predictions))
        #print('Job %s: Predicted row: %s' % (job_id, str(throughputs_matrix[-1])))
        #print('Job type %s oracle row: %s' % (self._jobs[job_id].job_type, self._reference_throughputs[worker_type][list(self._reference_job_types).index(self._jobs[job_id].job_type)]))
        #print('Original job type: %s' % (self._jobs[job_id].job_type))
        #print('Predicted job type: %s' % (predicted_job_type))

        # Set the throughputs using the oracle throughputs given by the
        # reference models.
        for other_job_id in self._jobs:
            if (job_id == other_job_id or
                other_job_id not in self._reference_job_map):
                continue
            reference_job_types = []
            reference_isolated_throughputs = []
            merged_job_id = job_id_pair.JobIdPair(job_id[0], other_job_id[0])
            true_isolated_throughputs = []
            for i, single_job_id in enumerate(merged_job_id.singletons()):
                true_isolated_throughputs.append(
                    self._throughputs[single_job_id][worker_type])
                reference_job_type = self._reference_job_map[single_job_id]
                reference_job_types.append(reference_job_type)
                reference_isolated_throughputs.append(
                    oracle_throughputs[reference_job_type]['null'])
            reference_colocated_throughputs = \
                oracle_throughputs[reference_job_types[0]][reference_job_types[1]]
            reference_normalized_throughputs = \
                np.divide(reference_colocated_throughputs,
                          reference_isolated_throughputs)
            self._throughputs[merged_job_id][worker_type] = \
                np.multiply(true_isolated_throughputs,
                            reference_normalized_throughputs)

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _reset_time_run_so_far(self):
        """Reset _time_run_so_far so that all jobs receive new fair allocation
        from here on out.

        Requires self._scheduler_lock to be held when calling this function.
        """
        current_time = self.get_current_timestamp()
        elapsed_time_since_last_reset = current_time - self._last_reset_time
        for worker_type in self._worker_types:
            self._worker_time_so_far[worker_type] = 0.0
            for job_id in self._job_time_so_far:
                # _job_time_so_far keeps track of how long job_id has run on
                # worker_type since the last reset event.
                if worker_type not in self._job_time_so_far[job_id]:
                    time_received = 0.0
                else:
                    # Ignore the initial time recorded for the job.
                    time_received = \
                            (self._job_time_so_far[job_id][worker_type] -
                             (self._time_per_iteration / 2.0))

                # Compute the time this job_id should have received since the
                # last reset event.
                if self._allocation is None or job_id not in self._allocation:
                    time_should_have_received = 0
                else:
                    time_should_have_received = \
                            self._allocation[job_id][worker_type] *\
                                elapsed_time_since_last_reset

                # deficit is now just the difference between the time job_id
                # should have received, and how much it actually received.
                deficit = time_should_have_received - time_received
                if job_id not in self._deficits[worker_type]:
                    self._deficits[worker_type][job_id] = 0.0
                self._deficits[worker_type][job_id] += deficit

                self._job_time_so_far[job_id][worker_type] = \
                        (self._time_per_iteration / 2.0)
                self._worker_time_so_far[worker_type] += \
                        self._job_time_so_far[job_id][worker_type]
        # Prints deficits every time allocation is reset.
        # self._print_deficits()
        self._last_reset_time = current_time

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _add_to_priorities(self, job_id, worker_type=None):
        """Adds a job_id to each worker type's priority list.
        NOTE: Used when scheduling is performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' priority data structures.
        """

        worker_types = self._worker_types
        if worker_type is not None:
            worker_types = [worker_type]
        for worker_type in worker_types:
            self._priorities[worker_type][job_id] = 0.0
            self._deficits[worker_type][job_id] = 0.0
            for other_job_id in self._throughputs:
                if (other_job_id.is_pair() and
                    job_id.overlaps_with(other_job_id)):
                    self._priorities[worker_type][other_job_id] = 0.0
                    self._deficits[worker_type][other_job_id] = 0.0

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _remove_from_priorities(self, job_id):
        """Removes a job_id from each worker type's priority list.
        NOTE: Used when scheduling is performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' priority data structures.
        """
        for worker_type in self._worker_types:
            while True:
                found = False
                for other_job_id in self._priorities[worker_type]:
                    if job_id.overlaps_with(other_job_id):
                        del self._priorities[worker_type][other_job_id]
                        del self._deficits[worker_type][other_job_id]
                        found = True
                        break
                if not found:
                    break

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _update_priorities(self):
        """Updates each per-worker priority data structure.

        Re-sorts the data structure of each worker to compute the next job to run.
        For a given worker w_i, the next job to be scheduled will be the job
        that has so far received the smallest fraction of its computed
        fair allocation.
        Requires self._scheduler_lock to be held when calling this function.

        NOTE: Used when scheduling is performed in rounds.
        """

        if self._need_to_update_allocation:
            self._reset_time_run_so_far()
            if self._estimate_throughputs:
                for worker_type in self._jobs_to_profile:
                    for job_id in self._profiled_jobs[worker_type]:
                        if not job_id in self._jobs:
                            continue
                        self._match_job_to_reference_job(job_id, worker_type)
                    self._profiled_jobs[worker_type] = {}
            self._allocation = self._get_allocation()
            self._need_to_update_allocation = False

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            for job_id in self._job_time_so_far:
                fraction = self._job_time_so_far[job_id][worker_type] / \
                         self._worker_time_so_far[worker_type]
                fractions[worker_type][job_id] = fraction
            for job_id in self._priorities[worker_type]:
                # Don't use inf so 2*new_priority > new_priority.
                #
                # Scale the default value by the allocation so that newly
                # added jobs run according to their respective allocations.
                new_priority = self._allocation[job_id][worker_type] * 1e9
                if self._allocation[job_id][worker_type] == 0.0:
                    assert(new_priority == 0)
                elif ((job_id.is_pair() and
                       (self._throughputs[job_id][worker_type][0] == 0 or
                        self._throughputs[job_id][worker_type][1] == 0)) or
                      (not job_id.is_pair() and
                       self._throughputs[job_id][worker_type] == 0)):
                    new_priority = 0
                elif fractions[worker_type][job_id] > 0.0:
                    new_priority = self._allocation[job_id][worker_type] /\
                            fractions[worker_type][job_id]
                self._priorities[worker_type][job_id] = new_priority

    def _add_available_worker_id(self, worker_id):
        """Adds a worker_id to the list of available workers."""

        self._available_worker_ids.put(worker_id)

    def _remove_available_worker_id(self, worker_id=None):
        """Returns the worker_id of the next available worker."""

        if self._simulate:
            try:
                return self._available_worker_ids.get_nowait()
            except queue.Empty as e:
                return None
        else:
            return self._available_worker_ids.get()

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _get_remaining_steps(self, job_id):
        steps_run_so_far = self._total_steps_run[job_id]
        return self._jobs[job_id].total_steps - steps_run_so_far

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def get_current_timestamp(self):
        if self._simulate:
            return self._current_timestamp
        else:
            return time.time()

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(self, worker_type, ip_addr=None, port=None):
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
            self._cumulative_worker_time_so_far[worker_id] = 0.0
            found = True
            if worker_type not in self._worker_type_to_worker_id_mapping:
                found = False
                self._worker_type_to_worker_id_mapping[worker_type] = []
            self._worker_type_to_worker_id_mapping[worker_type].append(worker_id)

            if not found:
                self._priorities[worker_type] = {}
                self._deficits[worker_type] = {}
                if self._estimate_throughputs:
                    self._jobs_to_profile[worker_type] = set()
                    self._profiled_jobs[worker_type] = {}
                for job_id in self._jobs:
                    self._steps_run_so_far[job_id][worker_type] = 0
                    self._job_time_so_far[job_id][worker_type] = \
                            (self._time_per_iteration / 2.0)
                    self._set_initial_throughput(job_id, worker_type)
                    if self._job_packing:
                        self._populate_job_combination_metadata(job_id,
                                                                worker_type)
                    self._initialize_num_steps_per_iteration(job_id, worker_type)
                    # Add to relevant priority data structure.
                    self._add_to_priorities(job_id, worker_type=worker_type)
                    if self._throughput_estimation:
                        self._jobs_to_profile[worker_type][job_id] = set()
                if worker_type not in self._worker_time_so_far:
                    self._worker_time_so_far[worker_type] = 0.0

            self._add_available_worker_id(worker_id)

            if worker_type not in self._cluster_spec:
                self._cluster_spec[worker_type] = 0
            self._cluster_spec[worker_type] += 1
            if not self._simulate:
                self._worker_connections[worker_id] = \
                    scheduler_client.SchedulerRpcClient(ip_addr, port)

            self._worker_start_times[worker_id] = self.get_current_timestamp()
            self._need_to_update_allocation = True

        return worker_id

    def _done_callback(self, job_id, worker_id, all_num_steps,
                       all_execution_times):
        """Handles completion of a scheduled job.

        Updates the running total of completed steps and time spent on each
        worker, for every currently active application. Removes the job from
        the scheduler if the job has finished all its requested steps. Adds
        the worker back to the list of available workers.

        Args:
            job_id: The id of the completed job(s).
            worker_id: The id of the worker where the job(s) were completed.
            all_num_steps: List of the number of steps each job ran for.
        """

        self._add_available_worker_id(worker_id)

        to_remove = []
        with self._scheduler_lock:
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            current_timestamp = self.get_current_timestamp()

            if np.min(all_execution_times) <= 0:
                # Micro-task failed.
                print(('%s]\t[Micro-task failed]\t'
                       'Job ID: %s') % (current_timestamp,
                                        job_id))
                if not job_id.is_pair():
                    self._num_failures_per_job[job_id] += 1
                    if (self._num_failures_per_job[job_id] >=
                        MAX_FAILED_ATTEMPTS):
                        print(('%s]\t[Job failed]\t'
                               'Job ID: %s') % (current_timestamp, job_id))
                        to_remove.append(job_id)

            else:
                print(('%s]\t[Micro-task succeeded]\t'
                       'Job ID: %s\tWorker type: %s\t'
                       'Worker ID: %d') % (current_timestamp,
                                           job_id,
                                           worker_type,
                                           worker_id))
                self._num_failures_per_job[job_id] = 0
                for single_job_id, num_steps, execution_time in \
                        zip(job_id.singletons(), all_num_steps,
                            all_execution_times):
                    # Job may be multi-GPU, and have already been removed from
                    # running_jobs by another worker.
                    if single_job_id in self._running_jobs:
                        self._running_jobs.remove(single_job_id)
                        self._steps_run_so_far[single_job_id][worker_type] += \
                                num_steps
                        self._total_steps_run[single_job_id] += num_steps
                        if (self._total_steps_run[single_job_id] <
                             self._jobs[single_job_id].total_steps):
                            pass
                        else:
                            finish_time = \
                                    self._per_job_latest_timestamps[single_job_id]
                            print(('%s]\t[Job succeeded]\t'
                                   'Job ID: %s') % (finish_time,
                                                    single_job_id))
                            to_remove.append(single_job_id)
                    if not self._simulate:
                        # NOTE: We update the timestamp before calling this
                        # function in simulation.
                        self._per_job_latest_timestamps[single_job_id] = \
                                self.get_current_timestamp()

                # If we just ran co-located jobs, use the maximum of the
                # individual execution times.
                max_execution_time = np.max(all_execution_times)
                # Job may be multi-GPU, and have already been marked complete
                # by another worker.
                # Divide by scale_factor so that _job_time_so_far and
                # _worker_time_so_far are incremented in total by
                # max_execution_time (_worker_time_so_far is just
                # _job_time_so_far summed over all possible job_ids).
                if job_id in self._job_time_so_far:
                    scale_factor = None
                    for single_job_id in job_id.singletons():
                        if single_job_id in self._jobs:
                            scale_factor = self._jobs[single_job_id].scale_factor
                    if scale_factor is not None:
                        self._job_time_so_far[job_id][worker_type] += \
                            (max_execution_time / scale_factor)
                        self._worker_time_so_far[worker_type] += \
                            (max_execution_time / scale_factor)
                self._cumulative_worker_time_so_far[worker_id] += \
                    max_execution_time

            self._update_throughput(job_id, worker_type,
                                    all_num_steps,
                                    all_execution_times)

        for single_job_id in to_remove:
            self.remove_job(single_job_id[0])
