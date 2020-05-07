from __future__ import print_function

import copy
import heapq
import numpy as np
import os
# from preconditions import preconditions
import queue
import threading
import time
import datetime
import random
import math
import matrix_completion
import warnings

# TODO: clean these up.
from job import Job
import job_id_pair
from job_table import JobTable
from runtime.rpc import scheduler_server, scheduler_client
import utils

SCHEDULER_PORT = 50060
SLEEP_SECONDS = 2
INFINITY = int(1e9)
DEFAULT_THROUGHPUT = 1
DEFAULT_NUM_STEPS = 100     # Default number of steps in each iteration.
EMA_ALPHA = .25 # Alpha parameter for exponential moving average.
MAX_FAILED_ATTEMPTS = 5
DEFAULT_MATRIX_COMPLETION_K = 10
DEFAULT_MATRIX_COMPLETION_MU = 1e-2

class Scheduler:

    # TODO: Make assign_SLOs a configurable parameter from scripts.
    def __init__(self, policy, simulate=False, throughputs_file=None,
                 seed=0, time_per_iteration=1920, profiling_percentage=0.0,
                 num_reference_models=16,
                 per_instance_type_prices_dir=None,
                 available_clouds=[],
                 assign_SLOs=False,
                 enable_global_queue=False,
                 expected_num_workers=None):


        # Print config information.
        if simulate:
            print('Running scheduler in simulation with the following args:')
        else:
            print('Running scheduler at %s:%s with the '
                  'following args:' % (utils.get_ip_address(), SCHEDULER_PORT))
        print('policy=%s' % (policy.name))
        print('seed=%d' % (seed))
        print('time_per_iteration=%d' % (time_per_iteration))
        print('profiling_percentage=%f' % (profiling_percentage))
        print('num_reference_models=%d' % (num_reference_models))

        # Flag to control whether scheduler runs in simulation mode.
        self._simulate = simulate
        # Initialize seeds.
        self._initialize_seeds(seed)
        # Initialize time in seconds each iteration should run for.
        self._time_per_iteration = time_per_iteration

        # Sets whether to use a global queue across all worker types.
        self._enable_global_queue = enable_global_queue

        self._expected_num_workers = expected_num_workers

        if self._simulate:
            self._start_timestamp = 0
        else:
            self._start_timestamp = time.time()
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
        # Total cost of each job so far.
        self._job_cost_so_far = {}
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
        # Throughputs measured with respect to job types rather than
        # individual jobs.
        # TODO: Use this to replace self._throughputs.
        self._job_type_throughputs = {}
        # Map from job ID to application.
        self._job_id_to_job_type = {}
        # Map from application to set of job IDs.
        self._job_type_to_job_ids = {}
        # Throughputs for all job types (pre-measured).
        if throughputs_file is not None:
            self._oracle_throughputs = utils.read_all_throughputs_json_v2(
                throughputs_file)
        else:
            self._oracle_throughputs = None
        # Flag to indicate whether throughputs should be estimated online.
        self._estimate_throughputs = \
            self._job_packing and profiling_percentage > 0
        if per_instance_type_prices_dir is not None:
            self._per_instance_type_spot_prices = \
                utils.read_per_instance_type_spot_prices_json(
                    per_instance_type_prices_dir)
            self._per_worker_type_prices = {}
            self._available_clouds = set(available_clouds)
            if assign_SLOs:
                self._SLOs = {}
            else:
                self._SLOs = None
        else:
            self._SLOs = None
            self._per_instance_type_spot_prices = None
            self._per_worker_type_prices = None
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
            # Sets the number of profiling data points to measure
            # for each newly arrived job.
            self._num_profiling_data_points_per_job = \
                int(num_reference_models * .6)
        # The per-round maximum number of steps to run for distributed jobs.
        self._max_steps = {}
        # All per-round lease update requests for distributed jobs.
        self._lease_update_requests = {}
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
            'UpdateLease': self._update_lease_callback,
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

        self._SLO_generator = random.Random()
        self._SLO_generator.seed(seed+6)

    def start_scheduling_thread(self):
        self.scheduler_thread = threading.Thread(
            target=self.schedule,
            args=())
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()


    def _update_per_worker_type_prices(self):
        assert(self._per_worker_type_prices is not None)
        current_time = self.get_current_timestamp(in_seconds=True)
        for worker_type in self._per_worker_type_prices:
            latest_price = \
                utils.get_latest_price_for_worker_type(
                    worker_type, current_time,
                    self._per_instance_type_spot_prices,
                    self._available_clouds)
            if self._per_worker_type_prices[worker_type] != latest_price:
                self._per_worker_type_prices[worker_type] = latest_price
                self._need_to_update_allocation = True

    def _update_throughput(self, job_id, worker_type, all_num_steps,
                           all_execution_times):
        # Job might have already completed.
        if job_id not in self._jobs:
            return
        if self._simulate and self._estimate_throughputs:
            if not job_id.is_pair():
                # Assume single job throughputs are already populated.
                return
            elif (job_id.is_pair() and
                  not self._throughputs_mask[job_id][worker_type]):
                self._throughputs_mask[job_id][worker_type] = True
                oracle_throughputs = self._oracle_throughputs[worker_type]
                scale_factor = self._jobs[job_id[0]].scale_factor
                job_types = []
                for single_job_id in job_id.singletons():
                    job_types.append((self._jobs[single_job_id].job_type,
                                      scale_factor))
                self._throughputs[job_id][worker_type] = \
                    oracle_throughputs[job_types[0]][job_types[1]]
                self._throughputs[job_id][worker_type] = \
                    [x / scale_factor for x in \
                        self._throughputs[job_id][worker_type]]
        elif not self._simulate:
            # Adjust the job throughput using an exponential moving average
            # between the old value and the new measurement.
            if job_id.is_pair():
                old_throughput = \
                    copy.deepcopy(self._throughputs[job_id][worker_type])
            else:
                old_throughput = [self._throughputs[job_id][worker_type]]

            for i, single_job_id in enumerate(job_id.singletons()):
                if all_execution_times[i] <= 0:
                    new_throughput = 0
                else:
                    new_throughput = all_num_steps[i] / all_execution_times[i]
                if old_throughput != INFINITY:
                    new_throughput *= EMA_ALPHA
                    new_throughput += (1 - EMA_ALPHA) * old_throughput[i]
                if job_id.is_pair():
                    self._throughputs[job_id][worker_type][i] =\
                        new_throughput
                else:
                    self._throughputs[job_id][worker_type] = new_throughput
            # Manually set failed job pair throughputs to 0.
            if np.min(all_execution_times) <= 0:
                if job_id.is_pair():
                    self._throughputs[job_id][worker_type] = [0.0, 0.0]
            new_throughput_str =\
                str(self._throughputs[job_id][worker_type])
            print(('[DEBUG] Job %s throughput on worker type %s: '
                   '%s -> %s') % (job_id, worker_type, str(old_throughput),
                                  new_throughput_str))

    def _read_throughputs_for_job_type(self, job_type):
        self._job_type_throughputs[job_type] = {}
        other_job_types = list(self._job_type_throughputs.keys())
        for worker_type in self._worker_types:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            self._job_type_throughputs[job_type][worker_type] = {}
            # TODO: Support scale factors > 1.
            self._job_type_throughputs[job_type][worker_type][None] = \
                oracle_throughputs[(job_type, 1)]['null']
            if self._job_packing:
                for other_job_type in other_job_types:
                    # TODO: Support scale factors > 1.
                    colocated_throughputs = \
                        oracle_throughputs[(job_type, 1)][(other_job_type, 1)]
                    self._job_type_throughputs[job_type][worker_type][other_job_type] = \
                        colocated_throughputs[0]
                    self._job_type_throughputs[other_job_type][worker_type][job_type] = \
                        colocated_throughputs[1]



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
            self._job_cost_so_far[job_id] = 0.0
            self._throughputs[job_id] = {}
            job_type = self._jobs[job_id].job_type
            self._job_id_to_job_type[job_id] = job_type
            if job_type not in self._job_type_throughputs:
                self._job_type_to_job_ids[job_type] = set()
                if self._estimate_throughputs:
                    # TODO: Support throughput estimation.
                    pass
                else:
                    self._read_throughputs_for_job_type(job_type)
            self._job_type_to_job_ids[job_type].add(job_id)
            self._num_failures_per_job[job_id] = 0
            self._total_steps_run[job_id] = 0
            if self._SLOs is not None:
                assert(job.duration is not None)
                assert(job.SLO is not None)
                self._SLOs[job_id] = \
                    (job.SLO * job.duration +
                     self.get_current_timestamp(in_seconds=True))
            for worker_type in self._worker_types:
                self._steps_run_so_far[job_id][worker_type] = 0
                self._set_initial_throughput(job_id, worker_type)
                if self._job_packing:
                    self._populate_job_combination_metadata(job_id,
                                                            worker_type)
                # Randomly select an order of reference jobs for each newly
                # arrived job to co-locate with.
                if self._estimate_throughputs:
                    self._jobs_to_profile[worker_type][job_id] = \
                        self._throughput_estimation_generator.choice(
                                self._reference_job_types,
                                len(self._reference_job_types),
                                replace=False).tolist()
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
            self._job_priority_weights[job_id] = \
                self._jobs[job_id].priority_weight
            del self._jobs[job_id]
            if self._num_failures_per_job[job_id] >= MAX_FAILED_ATTEMPTS:
                print("Job %d failed\n\tStart timestamp: %.2f\n\t"
                      "End timestamp: %.2f\nDuration: %.2f %s\n"
                      "Number of remaining active jobs: %d\n" % (
                          job_id[0],
                          self._per_job_start_timestamps[job_id],
                          self._per_job_latest_timestamps[job_id],
                          duration, "seconds", len(self._jobs))
                      )
                self._job_completion_times[job_id] = None
            else:
                print("Job %d completed\n\tStart timestamp: %.2f\n\t"
                  "End timestamp: %.2f\nDuration: %.2f %s\n"
                  "Number of remaining active jobs: %d\n" % (
                      job_id[0],
                      self._per_job_start_timestamps[job_id],
                      self._per_job_latest_timestamps[job_id],
                      duration, "seconds", len(self._jobs))
                  )
                self._job_completion_times[job_id] = duration
            job_type = self._job_id_to_job_type[job_id]
            self._job_type_to_job_ids[job_type].remove(job_id)
            del self._steps_run_so_far[job_id]
            del self._total_steps_run[job_id]
            del self._job_time_so_far[job_id]
            del self._throughputs[job_id]
            del self._job_id_to_job_type[job_id]
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
                if len(self._job_type_to_job_ids[job_type]) == 0:
                    del self._job_type_to_job_ids[job_type]
                    del self._job_type_throughputs[job_type]
                    for other_job_type in self._job_type_throughputs:
                        for worker_type in self._job_type_throughputs[other_job_type]:
                            del self._job_type_throughputs[other_job_type][worker_type][job_type]
            if self._estimate_throughputs:
                for worker_type in self._profiled_jobs:
                    if job_id in self._jobs_to_profile[worker_type]:
                        del self._jobs_to_profile[worker_type][job_id]
                    if job_id in self._profiled_jobs[worker_type]:
                        del self._profiled_jobs[worker_type][job_id]
            self._remove_from_priorities(job_id)
            # TODO: Add a flag to choose whether to update allocation here.
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
        oracle_throughputs = self._oracle_throughputs[worker_type]

        # Fill in all profiling slots by assigning them to jobs in FIFO order.
        jobs_to_profile = sorted(self._jobs_to_profile[worker_type].keys())
        for job_id in jobs_to_profile:
            if job_id not in self._profiled_jobs[worker_type]:
                self._profiled_jobs[worker_type][job_id] = {}
            while (used_profiling_slots < available_profiling_slots):
                job_type = (self._jobs[job_id].job_type, 1)
                reference_job_type = \
                    self._jobs_to_profile[worker_type][job_id].pop(0)
                isolated_throughputs = []
                isolated_throughputs.append(
                        oracle_throughputs[job_type]['null'])
                isolated_throughputs.append(
                        oracle_throughputs[reference_job_type]['null'])
                self._profiled_jobs[worker_type][job_id][reference_job_type] = \
                    np.divide(oracle_throughputs[job_type][reference_job_type],
                              isolated_throughputs)
                used_profiling_slots += 1
                if (len(self._profiled_jobs[worker_type][job_id]) ==
                    self._num_profiling_data_points_per_job):
                    del self._jobs_to_profile[worker_type][job_id]
                    break
        num_workers_left -= int(math.ceil(used_profiling_slots / \
                                          num_profiling_jobs_per_machine))
        return num_workers_left


    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper(self, worker_order):
        """Greedily selects the jobs to run in the next round by iterating
           through the job list in sorted priority order.

           Assumes only single-GPU jobs.

           Returns:
             A list of job IDs to schedule on the passed-in worker_type in
             the upcoming round.
        """

        already_scheduled_jobs = set()
        scheduled_jobs = {}

        num_workers_left = {}
        for worker_type in self._worker_types:
            scheduled_jobs[worker_type] = []
            num_workers = self._cluster_spec[worker_type]
            if self._estimate_throughputs:
                num_workers_left[worker_type] = \
                    self._profile_with_reference_jobs(worker_type,
                                                      num_workers)
            else:
                num_workers_left[worker_type] = num_workers

        sorted_job_queue = []
        for worker_type in worker_order:
            per_worker_type_entries = []
            for job_id in self._priorities[worker_type]:
                per_worker_type_entries.append(
                        (job_id, worker_type,
                         self._priorities[worker_type][job_id],
                         self._deficits[worker_type][job_id],
                         self._allocation[job_id][worker_type]))
            if not self._enable_global_queue:
                sorted_job_queue += sorted(per_worker_type_entries,
                                           key=lambda x: (x[2], x[3], x[4]),
                                           reverse=True)
            else:
                sorted_job_queue += per_worker_type_entries

        if self._enable_global_queue:
            sorted_job_queue.sort(key=lambda x: (x[2], x[3], x[4]),
                                  reverse=True)

        for job_id, worker_type, *_ in sorted_job_queue:
            if num_workers_left[worker_type] == 0:
                continue
            # Don't schedule jobs that have already been scheduled.
            if ((not job_id.is_pair() and job_id in already_scheduled_jobs) or
                (job_id.is_pair() and
                 (job_id.singletons()[0] in already_scheduled_jobs or
                  job_id.singletons()[1] in already_scheduled_jobs))):
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
            if scale_factor > num_workers_left[worker_type]:
                continue
            num_workers_left[worker_type] -= scale_factor

            for single_job_id in job_id.singletons():
                already_scheduled_jobs.add(single_job_id)
            scheduled_jobs[worker_type].append((job_id, scale_factor))

        return scheduled_jobs


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

        to_remove = []
        worker_types = ["v100", "p100", "k80"]

        for i, worker_type in enumerate(worker_types):
            if worker_type not in self._worker_type_to_worker_id_mapping:
                to_remove.append(i)
        for i in reversed(to_remove):
            worker_types.pop(i)

        if ('Perf' not in self._policy.name and
            'Packing' not in self._policy.name):
            self._worker_type_shuffler.shuffle(worker_types)

        worker_assignments = []
        scheduled_jobs = self._schedule_jobs_on_workers_helper(worker_types)

        for worker_type in worker_types:
            # Worker IDs organized into servers.
            worker_ids = copy.copy(
                self._worker_type_to_worker_id_mapping[worker_type])
            server_id_ptr = 0
            # Sort jobs by the scale factor: want to assign jobs from largest to smallest
            # scale factor to minimize fragmentation.
            scheduled_jobs[worker_type].sort(key=lambda x: x[1], reverse=True)
            num_workers_assigned = 0

            for (job_id, scale_factor) in scheduled_jobs[worker_type]:

                # Assign workers to jobs. Assign workers in a strided fashion to
                # minimize the number of servers used.
                worker_ids_for_job = []
                while len(worker_ids_for_job) < scale_factor:
                    num_workers = min(len(worker_ids[server_id_ptr]),
                                      scale_factor - len(worker_ids_for_job))
                    worker_ids_for_job.extend(worker_ids[server_id_ptr][:num_workers])
                    worker_ids[server_id_ptr] = worker_ids[server_id_ptr][num_workers:]
                    server_id_ptr += 1
                    server_id_ptr = server_id_ptr % len(worker_ids)
                worker_assignments.append(
                        (job_id,
                         tuple(worker_ids_for_job)))
                num_workers_assigned += scale_factor

                for single_job_id in job_id.singletons():
                    """
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
                    """
                    num_steps = self._jobs[single_job_id].total_steps
                    self._per_job_latest_timestamps[single_job_id] = \
                        self.get_current_timestamp()
                    self._running_jobs.add(single_job_id)
                worker_types = []
                for x in self._allocation[job_id]:
                    worker_types.append(x)
                worker_types = sorted(worker_types)
                allocation_str = ''
                for x in worker_types:
                    allocation_str += \
                        ' [%4s %f]' % (x, self._allocation[job_id][x])
                print(('%s]\t[Micro-task scheduled]\tJob ID: %s\t'
                       'Worker type: %s\tWorker ID(s): %s\t'
                       'Priority: %f\tDeficit: %f\t'
                       'Allocation: %s') % (self.get_current_timestamp(),
                                           job_id, worker_type,
                                           ",".join(["%d" % x
                                                     for x in worker_ids_for_job]),
                                           self._priorities[worker_type][job_id],
                                           self._deficits[worker_type][job_id],
                                           allocation_str))
            if num_workers_assigned < self._cluster_spec[worker_type]:
                print(('WARNING: %d GPUs of type %s left unused. '
                       'Number of active jobs: %d') % (
                    self._cluster_spec[worker_type] - num_workers_assigned,
                    worker_type,
                    len(self._jobs)))

        return worker_assignments

    def _get_num_steps(self, job_id, worker_type, single_job_id=None):
        if self._simulate:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            if job_id.is_pair():
                assert(single_job_id is not None)
                index = job_id.as_tuple().index(single_job_id[0])
                scale_factor = self._jobs[single_job_id].scale_factor
                job_types = []
                for x in job_id.singletons():
                    job_types.append((self._jobs[x].job_type, scale_factor))
                colocated_throughputs = \
                    oracle_throughputs[job_types[0]][job_types[1]]
                colocated_throughputs = \
                    [x / scale_factor for x in colocated_throughputs]
                single_job_throughput = colocated_throughputs[index]
                num_steps = int(single_job_throughput *
                                self._time_per_iteration)
            else:
                # NOTE: Assumes oracle throughputs for single jobs.
                num_steps = int(self._throughputs[job_id][worker_type] *
                                self._time_per_iteration)
        else:
            if job_id.is_pair():
                assert(single_job_id is not None)
                index = job_id.as_tuple().index(single_job_id[0])
                num_steps = int(self._throughputs[job_id][worker_type][index] *
                                self._time_per_iteration)
            else:
                num_steps = int(self._throughputs[job_id][worker_type] *
                                self._time_per_iteration)

        if single_job_id is not None:
            return min(num_steps,
                       self._get_remaining_steps(single_job_id))
        else:
            return min(num_steps,
                       self._get_remaining_steps(job_id))

    def _get_job_steps_and_finish_times(self, job_id, worker_type):
        """Returns the number of steps to execute and and latest finish time(s)
           for a job or job pair."""
        max_finish_time = self.get_current_timestamp()
        all_num_steps = []
        single_job_ids = job_id.singletons()
        if job_id.is_pair() and self._estimate_throughputs and self._simulate:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            scale_factor = self._scale_factors[job_id[0]]
            job_types = []
            for single_job_id in single_job_ids:
                job_types.append((self._jobs[single_job_id].job_type,
                                  scale_factor))
            oracle_throughput =\
                oracle_throughputs[job_types[0]][job_types[1]]
            oracle_throughput = [x / scale_factor for x in oracle_throughput]
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
        assert(run_time > 0)
        if job_template.needs_data_dir:
            command = job_template.command % (run_dir, run_dir)
        else:
            command = job_template.command % (run_dir)

        scale_factor = 1
        # Copies Philly distribution.
        if generate_multi_gpu_jobs and job_template.distributed:
            r = self._job_generator.uniform(0, 1)
            if 0.7 <= r <= 0.8:
                scale_factor = 2
            elif 0.8 <= r <= 0.95:
                scale_factor = 4
            elif 0.95 <= r:
                scale_factor = 8
        num_steps = \
            (run_time *
             self._oracle_throughputs['v100'][(job_type, scale_factor)]['null'])
        assert(num_steps > 0)

        priority_weight = 1.0
        if generate_multi_priority_jobs:
            r = self._job_generator.uniform(0, 1)
            if 0.0 <= r <= 0.2:
                priority_weight = 5.0

        SLO = None
        if self._SLOs is not None:
            r = self._SLO_generator.uniform(0, 1)
            if 0.0 <= r < 0.33:
                SLO = 1.2
            elif 0.33 <= r < 0.67:
                SLO = 2.0
            else:
                SLO = 10.0

        job = Job(job_id=None,
                  job_type=job_type,
                  command=command,
                  num_steps_arg=job_template.num_steps_arg,
                  total_steps=num_steps,
                  duration=run_time,
                  scale_factor=scale_factor,
                  priority_weight=priority_weight,
                  SLO=SLO)

        return job

    def simulate(self, cluster_spec, arrival_times=None, jobs=None,
                 measure_steady_state_jobs=False, lam=None,
                 jobs_to_complete=None,
                 fixed_job_duration=None, num_total_jobs=None,
                 generate_multi_gpu_jobs=False,
                 generate_multi_priority_jobs=False,
                 simulate_steady_state=False, debug=False,
                 checkpoint_threshold=None,
                 checkpoint_file=None,
                 num_gpus_per_server=None):
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
        if arrival_times is not None and len(arrival_times) > 0:
            next_job_arrival_time = arrival_times[0]
        no_dispatched_or_running_jobs = False
        current_round_start_time = 0
        current_round_end_time = None
        num_completed_jobs = 0

        # Set up the cluster according to the provided spec.
        worker_types = sorted([worker_type for worker_type in cluster_spec])
        for worker_type in worker_types:
            num_gpus = 1
            if num_gpus_per_server is not None:
                num_gpus = num_gpus_per_server[worker_type]
            for i in range(cluster_spec[worker_type] // num_gpus):
                self._register_worker_callback(worker_type,
                                               num_gpus=num_gpus)

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
            self._current_timestamp = arrival_times[0]
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
            if jobs_to_complete is not None:
                print("Number of completed jobs: %d" % len(jobs_to_complete.intersection(completed_jobs)))
                if jobs_to_complete.issubset(completed_jobs):
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

            # Update per-instance type prices.
            if self._per_worker_type_prices is not None:
                self._update_per_worker_type_prices()

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
                        if num_jobs_generated >= num_total_jobs:
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
            time.sleep(5)
            with self._scheduler_lock:
                num_workers = len(self._worker_ids)
                num_jobs = len(self._jobs)
                if num_workers == 0 or num_jobs == 0:
                    continue
                elif (self._expected_num_workers is not None and
                      num_workers < self._expected_num_workers):
                    # Hack to allow scheduler to wait for all workers to be
                    # launched before starting to dispatch jobs.
                    continue
                # Reset available_worker_ids to the desired size.
                self._available_worker_ids = queue.Queue(num_workers)
                for worker_id in self._worker_ids:
                    self._add_available_worker_id(worker_id)
                scheduled_jobs = self._schedule_jobs_on_workers()
                for (job_id, worker_ids) in scheduled_jobs:
                    worker_type = \
                        self._worker_id_to_worker_type_mapping[worker_ids[0]]
                    scale_factor = len(worker_ids)
                    for i, worker_id in enumerate(worker_ids):
                        job_descriptions = []
                        for j, single_job_id in enumerate(job_id.singletons()):
                            num_steps = self._jobs[single_job_id].total_steps
                            command = self._jobs[single_job_id].command
                            # Add distributed args if necessary.
                            if scale_factor > 1:
                                master_addr = \
                                    self._worker_connections[worker_ids[0]].addr
                                master_port = \
                                    self._worker_connections[worker_ids[0]].port
                                command = ('%s --master_addr %s '
                                           '--master_port %d '
                                           '--world_size %d '
                                           '--rank %d' % (command,
                                                          master_addr,
                                                          master_port + 1 + j,
                                                          scale_factor, i))
                            job_descriptions.append(
                                    (single_job_id,
                                     command,
                                     self._jobs[single_job_id].needs_data_dir,
                                     self._jobs[single_job_id].num_steps_arg,
                                     num_steps))
                        # Reset lease update metadata.
                        self._lease_update_requests[job_id] = []
                        self._max_steps[job_id] = None
                        self._worker_connections[worker_id].run(
                                job_descriptions, worker_id)
                        self._remove_available_worker_id(worker_id)
            while not self._available_worker_ids.full():
                time.sleep(2)
                continue

    def schedule(self):
        """Schedules jobs on workers."""
        self._schedule_with_rounds()


    def get_average_jct(self, job_ids=None, verbose=True):
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
                job_ids = sorted(list(self._job_completion_times.keys()))
            print('Job completion times:')
            all_job_completion_times = []
            low_priority_job_completion_times = []
            high_priority_job_completion_times = []
            for job_id in job_ids:
                completion_time = self._job_completion_times[job_id]
                if completion_time is None:
                    continue
                else:
                    all_job_completion_times.append(completion_time)
                if self._job_priority_weights[job_id] == 1.0:
                    print('Job %s: %.3f' % (job_id, completion_time))
                    low_priority_job_completion_times.append(completion_time)
                else:
                    print('Job %s (high priority): %.3f' % (job_id,
                                                            completion_time))
                    high_priority_job_completion_times.append(completion_time)
            average_job_completion_time = np.mean(all_job_completion_times)
            if verbose:
                print('Average job completion time: '
                      '%.3f seconds' % (average_job_completion_time))
                if len(low_priority_job_completion_times) > 0:
                    average_low_pri_jct = \
                        np.mean(low_priority_job_completion_times)
                    print('Average job completion time (low priority): '
                          '%.3f seconds' % (average_low_pri_jct))
                if len(high_priority_job_completion_times) > 0:
                    average_high_pri_jct = \
                        np.mean(high_priority_job_completion_times)
                    print('Average job completion time (high priority): '
                          '%.3f seconds' % (average_high_pri_jct))
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

    def get_total_cost(self, verbose=True):
        total_cost = 0.0
        for job_id in self._job_cost_so_far:
            total_cost += self._job_cost_so_far[job_id]
        if verbose:
            print('Total cost: $%.2f' % (total_cost))
        return total_cost

    def get_num_SLO_violations(self, verbose=True):
        num_SLO_violations = 0
        if self._SLOs is not None:
            for job_id in self._SLOs:
                SLO = self._SLOs[job_id]
                completion_time = self._job_completion_times[job_id]
                if verbose:
                    print('%s: completion_time=%f, SLO=%f, '
                          'completion_time / SLO = %f' % (job_id,
                                                          completion_time,
                                                          SLO,
                                                          completion_time / SLO))
                if completion_time > SLO:
                    num_SLO_violations += 1
        if verbose:
            print('Number of SLO violations: %d' % (num_SLO_violations))
        return num_SLO_violations

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
        priority_weights = {
            job_id: self._jobs[job_id].priority_weight
            for job_id in self._jobs
        }
        num_steps_remaining = {
            job_id: self._get_remaining_steps(job_id)
            for job_id in self._jobs
        }
        times_since_start = {
            job_id: self.get_current_timestamp() - self._per_job_start_timestamps[job_id]
            for job_id in self._jobs
        }
        if self._policy.name == "AlloX":
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors,
                times_since_start, num_steps_remaining,
                self._cluster_spec)
        elif self._policy.name.startswith("FinishTimeFairness"):
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, priority_weights,
                times_since_start, num_steps_remaining,
                self._cluster_spec)
        elif self._policy.name == "Isolated":
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, self._cluster_spec)
        elif self._policy.name.startswith("MaxMinFairness"):
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, priority_weights,
                self._cluster_spec)
        elif self._policy.name.startswith("MinTotalDuration"):
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, scale_factors, num_steps_remaining,
                self._cluster_spec)
        elif self._policy.name.startswith('ThroughputNormalizedByCostSum'):
            if 'SLO' in self._policy.name:
                SLOs = {}
                if self._SLOs is not None:
                    for job_id in self._jobs:
                        SLOs[job_id] = \
                            (self._SLOs[job_id] -
                             self.get_current_timestamp(in_seconds=True))
                else:
                    num_steps_remaining = {}
                unflattened_allocation = self._policy.get_allocation(
                    self._throughputs, scale_factors, self._cluster_spec,
                    instance_costs=self._per_worker_type_prices,
                    SLOs=SLOs,
                    num_steps_remaining=num_steps_remaining)
            else:
                unflattened_allocation = self._policy.get_allocation(
                    self._throughputs, scale_factors, self._cluster_spec,
                    instance_costs=self._per_worker_type_prices)
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
                if (self._oracle_throughputs is None or
                    self._estimate_throughputs or
                    job.scale_factor != other_job.scale_factor):
                    self._throughputs[merged_job_id][worker_type] = [0.0, 0.0]
                    if self._estimate_throughputs:
                        self._throughputs_mask[merged_job_id][worker_type] = \
                            False
                else:
                    oracle_throughputs = self._oracle_throughputs[worker_type]
                    # The single-job IDs for job pairs are stored in sorted
                    # order so make sure the co-located throughputs match this
                    # order.
                    scale_factor = job.scale_factor
                    job_types = [(job.job_type, scale_factor),
                                 (other_job.job_type, scale_factor)]
                    if job_id < other_job_id:
                        self._throughputs[merged_job_id][worker_type] = \
                            oracle_throughputs[job_types[0]][job_types[1]]
                    else:
                        self._throughputs[merged_job_id][worker_type] = \
                            oracle_throughputs[job_types[1]][job_types[0]]
                    self._throughputs[merged_job_id][worker_type] = \
                        [x / scale_factor for x in \
                            self._throughputs[merged_job_id][worker_type]]

    def _set_initial_throughput(self, job_id, worker_type):
        assert(not job_id.is_pair())
        if self._oracle_throughputs is not None:
            job_type = self._jobs[job_id].job_type
            scale_factor = self._jobs[job_id].scale_factor
            key = (job_type, scale_factor)
            self._throughputs[job_id][worker_type] = \
                self._oracle_throughputs[worker_type][key]['null'] /\
                    scale_factor
        else:
            self._throughputs[job_id][worker_type] = DEFAULT_THROUGHPUT

    def _initialize_reference_throughputs(self, num_reference_models):
        self._reference_throughputs = {}
        all_worker_types = sorted(self._oracle_throughputs.keys())
        all_job_types = []
        for key in self._oracle_throughputs[all_worker_types[0]].keys():
            if key[1] == 1:
                all_job_types.append(key)
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
                    colocated_throughputs = \
                        np.divide(oracle_throughputs[job_type_0][job_type_1],
                                  isolated_throughputs)
                    self._reference_throughputs[worker_type][i][j] = \
                            colocated_throughputs[0]
                    self._reference_throughputs[worker_type][j][i] = \
                            colocated_throughputs[1]
            for i in range(num_reference_models):
                if np.linalg.norm(self._reference_throughputs[worker_type][i]) == 0:
                    self._reference_throughputs[worker_type][i] += 0.0001

    def _cosine_distance(self, a, b):
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _match_job_to_reference_job(self, job_id, worker_type):
        num_reference_job_types = len(self._reference_job_types)
        oracle_throughputs = self._oracle_throughputs[worker_type]
        reference_throughputs = self._reference_throughputs[worker_type]

        # Add a row to reference throughputs for the newly arrived job.
        throughputs_matrix = \
            np.concatenate((reference_throughputs,
                            np.zeros((1, num_reference_job_types),
                                     dtype=np.float32)),
                           axis=0)

        # Initialize the mask.
        mask = np.concatenate((np.ones((num_reference_job_types,
                                        num_reference_job_types),
                                       dtype=np.float32),
                               np.zeros((1, num_reference_job_types),
                                        dtype=np.float32)),
                              axis=0)

        # Fill in measured data points.
        for i, reference_job_type in enumerate(self._reference_job_types):
            if reference_job_type in self._profiled_jobs[worker_type][job_id]:
                throughputs_matrix[-1][i] = \
                    self._profiled_jobs[worker_type][job_id][reference_job_type][0]
                mask[-1][i] = 1

        # Run matrix completion algorithm if there are values to estimate.
        if (np.min(mask) == 0):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                k = DEFAULT_MATRIX_COMPLETION_K
                mu = DEFAULT_MATRIX_COMPLETION_MU
                try:
                    estimated_throughputs = \
                        np.clip(matrix_completion.pmf_solve(throughputs_matrix,
                                                            mask,
                                                            k=k,
                                                            mu=mu),
                                0.0, 1.0)
                    for i in range(len(mask)):
                        for j in range(len(mask[0])):
                            if not mask[i][j]:
                                throughputs_matrix[i][j] = \
                                    estimated_throughputs[i][j]
                except np.linalg.LinAlgError as e:
                    print('WARNING: could not estimate throughputs!')
                    print(e)
                    estimated_throughputs = None


        # Measure the distance from the new row to every other row and find
        # the row with the smallest distance.
        distances = []
        if np.linalg.norm(throughputs_matrix[-1]) == 0:
            return
        for i, reference_job_type in enumerate(self._reference_job_types):
            distance = self._cosine_distance(throughputs_matrix[i],
                                             throughputs_matrix[-1])
            distances.append((reference_job_type, distance))
        distances.sort(key=lambda x: x[1])
        predicted_job_type = distances[0][0]
        self._reference_job_map[job_id] = predicted_job_type

        # Set the throughputs using the oracle throughputs given by the
        # reference models.
        for other_job_id in self._jobs:
            if (job_id == other_job_id or
                other_job_id not in self._reference_job_map):
                continue
            reference_job_types = []
            merged_job_id = job_id_pair.JobIdPair(job_id[0], other_job_id[0])
            true_isolated_throughputs = []
            for i, single_job_id in enumerate(merged_job_id.singletons()):
                true_isolated_throughputs.append(
                    self._throughputs[single_job_id][worker_type])
                reference_job_type = self._reference_job_map[single_job_id]
                reference_job_types.append(
                    list(self._reference_job_types).index(reference_job_type))
            reference_normalized_throughputs = \
                [reference_throughputs[reference_job_types[0]][reference_job_types[1]],
                 reference_throughputs[reference_job_types[1]][reference_job_types[0]]]
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
                    profiled_job_ids = \
                        sorted(self._profiled_jobs[worker_type].keys())
                    for job_id in profiled_job_ids:
                        if not job_id in self._jobs:
                            del self._profiled_jobs[worker_type][job_id]
                        elif (len(self._profiled_jobs[worker_type][job_id]) >=
                            self._num_profiling_data_points_per_job):
                            self._match_job_to_reference_job(job_id,
                                                             worker_type)
                            del self._profiled_jobs[worker_type][job_id]
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
    def get_current_timestamp(self, in_seconds=False):
        if self._simulate:
            return self._current_timestamp
        else:
            if in_seconds:
                return time.time() - self._start_timestamp
            else:
                return time.time()

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(self, worker_type, num_gpus=1,
                                  ip_addr=None, port=None):
        """Registers a worker with the scheduler.

        Initializes state for a new worker and assigns it an id.
        The worker provides an IP address and port for its RPC server
        so that the scheduler can establish an RPC client for
        scheduler-to-worker communication. The worker also
        enumerates its available devices so that the scheduler
        can make fine-grained scheduling decisions.

        Args:
            worker_type: The type of GPU available on the worker.
            num_gpus: The number of GPUs available on the worker.
            ip_addr: IP address of the worker's RPC server.
            port: Port number for the worker's RPC server.
            devices: List of available devices on the worker.

        Returns:
            The worker_id of the newly registered worker.
        """

        # Share a single RPC client for each GPU on the worker.
        if not self._simulate:
            rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)

        with self._scheduler_lock:
            # Update relevant data structures if worker type was
            # previously unseen.
            found = True
            if worker_type not in self._worker_type_to_worker_id_mapping:
                found = False
                self._worker_type_to_worker_id_mapping[worker_type] = []

            if not found:
                self._priorities[worker_type] = {}
                self._deficits[worker_type] = {}
                if self._estimate_throughputs:
                    self._jobs_to_profile[worker_type] = {}
                    self._profiled_jobs[worker_type] = {}
                    # Randomly select an order of reference jobs for each
                    # job to co-locate with.
                    for job_id in self._jobs:
                        self._jobs_to_profile[worker_type][job_id] = \
                            self._throughput_estimation_generator.choice(
                                    self._reference_job_types,
                                    len(self._reference_job_types),
                                    replace=False).tolist()
                if self._per_worker_type_prices is not None:
                    self._per_worker_type_prices[worker_type] = \
                        utils.get_latest_price_for_worker_type(
                            worker_type,
                            self.get_current_timestamp(in_seconds=True),
                            self._per_instance_type_spot_prices,
                            self._available_clouds)
                for job_id in self._jobs:
                    self._steps_run_so_far[job_id][worker_type] = 0
                    self._job_time_so_far[job_id][worker_type] = \
                            (self._time_per_iteration / 2.0)
                    self._set_initial_throughput(job_id, worker_type)
                    if self._job_packing:
                        self._populate_job_combination_metadata(job_id,
                                                                worker_type)
                    # Add to relevant priority data structure.
                    self._add_to_priorities(job_id, worker_type=worker_type)
                    if self._estimate_throughputs:
                        self._jobs_to_profile[worker_type][job_id] = set()
                if worker_type not in self._worker_time_so_far:
                    self._worker_time_so_far[worker_type] = 0.0

            # Update relevant data structures for each GPU available
            # on the worker.
            per_worker_ids = []
            for i in range(num_gpus):
                worker_id = self._worker_id_counter
                per_worker_ids.append(worker_id)
                self._worker_ids.append(worker_id)
                self._worker_id_counter += 1
                self._worker_types.add(worker_type)
                self._cumulative_worker_time_so_far[worker_id] = 0.0

                self._worker_id_to_worker_type_mapping[worker_id] = worker_type
                self._add_available_worker_id(worker_id)

                if worker_type not in self._cluster_spec:
                    self._cluster_spec[worker_type] = 0
                self._cluster_spec[worker_type] += 1
                if not self._simulate:
                    self._worker_connections[worker_id] = rpc_client

                self._worker_start_times[worker_id] = self.get_current_timestamp()
            self._worker_type_to_worker_id_mapping[worker_type].append(per_worker_ids)
            self._need_to_update_allocation = True

        return (per_worker_ids, self._time_per_iteration)

    def _update_lease_callback(self, job_id, worker_id, steps, duration,
                               max_steps, max_duration):
        scale_factor = self._jobs[job_id].scale_factor
        if steps == 0 or duration == 0:
            return (INFINITY, self._time_per_iteration)
        elif scale_factor == 1:
            return (max_steps, max_duration)
        else:
            with self._scheduler_lock:
                update_id = len(self._lease_update_requests[job_id])
                self._lease_update_requests[job_id].append((steps, duration,
                                                            max_steps,
                                                            max_duration))
                if update_id == 0:
                    assert self._max_steps[job_id] is None

            # The first worker to request a lease update computes the new
            # lease for all workers.
            if update_id == 0:
                with self._scheduler_lock:
                    remaining_time = \
                        (self._time_per_iteration -
                         duration % self._time_per_iteration)
                    throughput = steps / duration
                    remaining_steps = max(1, int(remaining_time * throughput))
                    max_completed_steps = \
                        max([request[0] for request in \
                                self._lease_update_requests[job_id]])
                    self._max_steps[job_id] = \
                        max_completed_steps + remaining_steps
                    return (self._max_steps[job_id], INFINITY)
            else:
                # Wait for the first update to complete.
                while True:
                    with self._scheduler_lock:
                        max_steps = self._max_steps[job_id]
                        if max_steps is not None:
                            break
                    # TODO: Sleep for less time?
                    time.sleep(1)
                assert max_steps is not None
                return (max_steps, INFINITY)

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

        # TODO: Aggregate updates before processing distributed jobs?
        to_remove = []
        with self._scheduler_lock:
            current_timestamp = self.get_current_timestamp()
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]

            if np.min(all_execution_times) <= 0 or np.min(all_num_steps) == 0:
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
                self._need_to_update_allocation = True

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
                    if self._per_worker_type_prices is not None:
                        self._job_cost_so_far[single_job_id] += \
                            (self._per_worker_type_prices[worker_type] *
                             execution_time / 3600.0)
                    job_cost_so_far = \
                        self._job_cost_so_far[single_job_id]
                    print('Job %s cost so far: $%.2f' % (single_job_id,
                                                         job_cost_so_far))
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

        self._add_available_worker_id(worker_id)
