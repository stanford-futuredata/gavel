from __future__ import print_function

import collections
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
import set_queue
from throughput_estimator import ThroughputEstimator
import utils

""" Constants """
# Port for scheduler server.
SCHEDULER_PORT = 50060
# Proxy for infinity.
INFINITY = int(1e9)
# Default job throughput.
DEFAULT_THROUGHPUT = 1
# Default number of steps in each iteration.
DEFAULT_NUM_STEPS = 100
# Alpha parameter for exponential moving average.
EMA_ALPHA = .5
# Maximum number of times a job is allowed to fail before being dropped.
MAX_FAILED_ATTEMPTS = 5
# Fraction of the round to wait for before re-computing the schedule.
SCHEDULE_RECOMPUTE_FRACTION = 0.5

class Scheduler:

    # TODO: Make assign_SLOs a configurable parameter from scripts.
    def __init__(self, policy, simulate=False, throughputs_file=None,
                 seed=0, time_per_iteration=1920, profiling_percentage=1.0,
                 num_reference_models=len(JobTable),
                 per_instance_type_prices_dir=None,
                 available_clouds=[],
                 assign_SLOs=False,
                 enable_global_queue=False,
                 expected_num_workers=None,
                 minimum_time_between_allocation_resets=1920,
                 max_rounds=None):


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
        self._minimum_time_between_allocation_resets = \
            minimum_time_between_allocation_resets

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
        # Synchronization primitives to ensure thread-safe updates of
        # scheduler metadata.
        self._scheduler_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._scheduler_lock)
        # List of available worker IDs.
        self._available_worker_ids = set_queue.SetQueue()
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Current map from job combinations to assigned workers.
        self._current_worker_assignments = collections.OrderedDict()
        # Map of jobs to worker assignments for the upcoming round.
        self._next_worker_assignments = None
        # Set of jobs that have been dispatched to workers.
        self._dispatched_jobs = set()
        # Set of jobs with an extended lease for the upcoming round.
        self._jobs_with_extended_lease = set()
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
            (self._job_packing and
             (profiling_percentage < 1 or num_reference_models < len(JobTable)))
        if self._estimate_throughputs:
            self._throughput_estimator = \
                self._initialize_throughput_estimator(seed+4,
                                                      num_reference_models,
                                                      profiling_percentage)
            self._reference_throughputs = \
                self._throughput_estimator.get_reference_throughputs()
            self._reference_job_map = {}
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
        # The per-round maximum number of steps to run for distributed jobs.
        self._max_steps = {}
        # All per-round lease update requests for distributed jobs.
        self._lease_update_requests = {}
        # List of all RPC clients.
        self._all_rpc_clients = []
        # Port offsets from rank-0 nodes of distributed jobs.
        self._master_port_offsets = {}
        # Currently running jobs.
        self._running_jobs = set()
        # The timestamp when each worker entered the cluster.
        self._worker_start_times = {}
        # Verbose flag.
        self._verbose = False
        # Data structures for debugging.
        self._micro_tasks_per_job = {}
        self._all_jobs = []
        # Queue for printing log information in a thread-safe way.
        self._write_queue = queue.Queue()
        # In-progress updates for distributed jobs.
        self._in_progress_updates = {}
        # Set of completed job IDs.
        self._completed_jobs = set()
        # Maximum number of rounds.
        self._max_rounds = max_rounds
        # Number of completed rounds.
        self._num_completed_rounds = 0

        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'UpdateLease': self._update_lease_callback,
            'Done': self._done_callback,
        }

        if not self._simulate:
            self._logging_thread = threading.Thread(target=self._print_logs)
            self._logging_thread.daemon = True
            self._logging_thread.start()

            self._allocation_thread = \
                threading.Thread(target=self._allocation_thread)
            self._allocation_thread.daemon = True
            self._allocation_thread.start()

            self.server_thread = threading.Thread(
                target=scheduler_server.serve,
                args=(port, callbacks, self._write_queue))
            self.server_thread.daemon = True
            self.server_thread.start()

            self._mechanism_thread = \
                threading.Thread(target=self._schedule_with_rounds_async)
            self._mechanism_thread.daemon = True
            self._mechanism_thread.start()


    def _initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

        self._job_generator = random.Random()
        self._job_generator.seed(seed+2)

        self._interarrival_time_generator = random.Random()
        self._interarrival_time_generator.seed(seed+3)

        self._worker_type_shuffler = random.Random()
        self._worker_type_shuffler.seed(seed+5)

        self._SLO_generator = random.Random()
        self._SLO_generator.seed(seed+6)

    def _initialize_throughput_estimator(self, seed, num_reference_models,
                                         profiling_percentage):
        worker_types = []
        for worker_type in self._oracle_throughputs:
            if 'unconsolidated' not in worker_type:
                worker_types.append(worker_type)
        worker_types.sort()
        job_types = [(job_template.model, 1) for job_template in JobTable]
        return ThroughputEstimator(self._oracle_throughputs,
                                   worker_types, job_types,
                                   num_reference_models,
                                   profiling_percentage,
                                   seed)

    def _print_logs(self):
        while True:
            output = self._write_queue.get()
            print('[%s] %s' % (str(datetime.datetime.now()), output),
                  flush=True)

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
                self._scheduler_cv.acquire()
                self._need_to_update_allocation = True
                self._scheduler_cv.notify()
                self._scheduler_cv.release()

    def _update_throughput(self, job_id, worker_type, all_num_steps,
                           all_execution_times):
        # Job might have already completed.
        if not job_id.is_pair() and not job_id in self._jobs:
            return
        if self._simulate and self._estimate_throughputs:
            if not job_id.is_pair():
                # Assume single job throughputs are already populated.
                return
            else:
                oracle_throughputs = self._oracle_throughputs[worker_type]
                scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
                job_types = []
                for single_job_id in job_id.singletons():
                    job_types.append((self._jobs[single_job_id].job_type,
                                      scale_factor))
                self._throughputs[job_id][worker_type] = \
                    oracle_throughputs[job_types[0]][job_types[1]]
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

    def _read_throughputs_for_job_type(self, job_type_key):
        """Reads oracle throughputs for passed in job type.

           Args:
             job_type_key: A tuple of (model, scale_factor).
        """
        self._job_type_throughputs[job_type_key] = {}
        other_job_type_keys = list(self._job_type_throughputs.keys())
        for worker_type in self._worker_types:
            oracle_throughputs = self._oracle_throughputs[worker_type]
            self._job_type_throughputs[job_type_key][worker_type] = {}
            self._job_type_throughputs[job_type_key][worker_type][None] = \
                oracle_throughputs[job_type_key]['null']
            if self._job_packing:
                for other_job_type_key in other_job_type_keys:
                    # Don't store throughputs for jobs with different scale
                    # factors.
                    if other_job_type_key[1] != job_type_key[1]:
                        continue
                    colocated_throughputs = \
                        oracle_throughputs[job_type_key][other_job_type_key]
                    self._job_type_throughputs[job_type_key][worker_type][other_job_type_key] = \
                        colocated_throughputs[0]
                    self._job_type_throughputs[other_job_type_key][worker_type][job_type_key] = \
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
            scale_factor = job.scale_factor
            job_type_key = (job_type, scale_factor)
            self._job_id_to_job_type[job_id] = job_type_key
            if self._estimate_throughputs:
                self._reference_job_map[job_id] = \
                    self._throughput_estimator.match_job_to_reference_job(job_type_key)
            if job_type_key not in self._job_type_throughputs:
                self._job_type_to_job_ids[job_type_key] = set()
                self._read_throughputs_for_job_type(job_type_key)
            self._job_type_to_job_ids[job_type_key].add(job_id)
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
            self._scheduler_cv.notify()

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
            self._completed_jobs.add(job_id)
            duration = self._per_job_latest_timestamps[job_id] - \
                self._per_job_start_timestamps[job_id]
            self._job_priority_weights[job_id] = \
                self._jobs[job_id].priority_weight
            job_type = self._jobs[job_id].job_type
            scale_factor = self._jobs[job_id].scale_factor
            job_type_key = (job_type, scale_factor)
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
            job_type_key = self._job_id_to_job_type[job_id]
            self._job_type_to_job_ids[job_type_key].remove(job_id)
            del self._steps_run_so_far[job_id]
            del self._job_time_so_far[job_id]
            del self._throughputs[job_id]
            del self._job_id_to_job_type[job_id]
            del self._num_failures_per_job[job_id]
            if job_id in self._in_progress_updates:
                del self._in_progress_updates[job_id]
            if job_id in self._lease_update_requests:
                del self._lease_update_requests[job_id]
            if job_id in self._max_steps:
                del self._max_steps[job_id]
            if job_id in self._jobs_with_extended_lease:
                self._jobs_with_extended_lease.remove(job_id)
            if self._job_packing:
                to_delete = []
                for other_job_id in self._throughputs:
                    if (other_job_id.is_pair() and
                        job_id.overlaps_with(other_job_id)):
                        to_delete.append(other_job_id)
                for other_job_id in to_delete:
                    del self._throughputs[other_job_id]
                    del self._job_time_so_far[other_job_id]
                    if other_job_id in self._in_progress_updates:
                        del self._in_progress_updates[other_job_id]
                    if other_job_id in self._lease_update_requests:
                        del self._lease_update_requests[other_job_id]
                    if other_job_id in self._max_steps:
                        del self._max_steps[other_job_id]
                    if other_job_id in self._jobs_with_extended_lease:
                        self._jobs_with_extended_lease.remove(other_job_id)

                if len(self._job_type_to_job_ids[job_type_key]) == 0:
                    del self._job_type_to_job_ids[job_type_key]
                    del self._job_type_throughputs[job_type_key]
                    for other_job_type_key in self._job_type_throughputs:
                        for worker_type in self._job_type_throughputs[other_job_type_key]:
                            if job_type_key in self._job_type_throughputs[other_job_type_key][worker_type]:
                                del self._job_type_throughputs[other_job_type_key][worker_type][job_type_key]
            self._remove_from_priorities(job_id)
            # TODO: Add a flag to choose whether to update allocation here.
            self._need_to_update_allocation = True
            self._scheduler_cv.notify()

    def num_workers(self):
        """Returns the number of workers the scheduler is connected to."""

        n = 0
        with self._scheduler_lock:
            for worker_type in self._cluster_spec:
                n += self._cluster_spec[worker_type]
            return n

    def is_done(self, jobs_to_complete=None):
        """Returns whether the scheduler is done with all its assigned work."""
        with self._scheduler_lock:
            if (self._max_rounds is not None and
                self._num_completed_rounds >= self._max_rounds):
                return True
            elif jobs_to_complete is not None:
                return jobs_to_complete.issubset(self._completed_jobs)
            else:
                return False

    def reset_workers(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            for i, rpc_client in enumerate(self._all_rpc_clients):
                rpc_client.reset()

    def shutdown(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            for rpc_client in self._all_rpc_clients:
                rpc_client.shutdown()
        # TODO: Any other cleanup?

    """
    ======================================================================
       Scheduler's main schedule() and simulate() methods.
    ======================================================================
    """

    def _print_schedule_summary(self):
        worker_types = sorted(self._cluster_spec.keys())
        for job_id, worker_ids in self._current_worker_assignments.items():
            if job_id not in self._jobs:
                continue
            allocation_str = ''
            for x in worker_types:
                allocation_str += \
                    ' [%4s %f]' % (x, self._allocation[job_id][x])
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            print(('%s]\t[Micro-task scheduled]\tJob ID: %s\t'
                   'Worker type: %s\tWorker ID(s): %s\t'
                   'Priority: %f\tDeficit: %f\t'
                   'Allocation: %s') % (self.get_current_timestamp(),
                                       job_id, worker_type,
                                       ",".join([str(x) for x in worker_ids]),
                                       self._priorities[worker_type][job_id],
                                       self._deficits[worker_type][job_id],
                                       allocation_str))
        num_workers_assigned = {}
        for job_id, worker_ids in self._current_worker_assignments.items():
            if job_id not in self._jobs:
                continue
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            if worker_type not in num_workers_assigned:
                num_workers_assigned[worker_type] = 0
            num_workers_assigned[worker_type] += len(worker_ids)
        for worker_type in worker_types:
            if worker_type not in num_workers_assigned:
                num_workers_assigned[worker_type] = 0
            if (num_workers_assigned[worker_type] <
                self._cluster_spec[worker_type]):
                unused_workers = (self._cluster_spec[worker_type] -
                                  num_workers_assigned[worker_type])
                print(('WARNING: %d GPUs of type %s left unused. '
                       'Number of active jobs: %d') % (unused_workers,
                                                       worker_type,
                                                       len(self._jobs)))

    def _assign_workers_to_job(self, job_id, scale_factor, worker_type,
                               worker_state, worker_assignments,
                               worker_ids_for_job=None):
        """Assign workers to jobs.

        Assigns workers in a strided fashion to minimize the number
        of servers used.

        Args:
          job_id: The job (combination) ID to schedule.
          scale_factor: The number of GPUs requested.
          worker_type: The worker type to allocate.
          worker_state: A dict comprised of the following information:
            worker_ids: Worker IDs organized into servers.
            reserved_worker_ids: A set of worker IDs that are already in use.
            server_id_ptr: The server to assign workers from.
            num_workers_assigned: The total number of allocated workers.
          worker_assignments: A list of (job_id, worker_ids) assignment tuples.
          worker_ids_for_job: An optional list of worker IDs to assign.
        """
        worker_ids = worker_state['worker_ids']
        reserved_worker_ids = worker_state['reserved_worker_ids']
        server_id_ptr = worker_state['server_id_ptr']
        num_workers_assigned = worker_state['num_workers_assigned']

        if worker_ids_for_job is None:
            worker_ids_for_job = []
            while len(worker_ids_for_job) < scale_factor:
                num_workers = min(len(worker_ids[server_id_ptr]),
                                  scale_factor - len(worker_ids_for_job))
                worker_ids_to_assign = worker_ids[server_id_ptr][:num_workers]
                ineligible_worker_ids = \
                    set(worker_ids_to_assign).intersection(reserved_worker_ids)
                # Only assign the worker IDs if they have not been reserved
                # for a different job.
                if len(ineligible_worker_ids) == 0:
                    worker_ids_for_job.extend(worker_ids_to_assign)
                # Update metadata regardless of whether the worker IDs were
                # assigned to this job; a different job could have reserved
                # these workers.
                worker_ids[server_id_ptr] = \
                    worker_ids[server_id_ptr][num_workers:]
                server_id_ptr += 1
                server_id_ptr = server_id_ptr % len(worker_ids)
        worker_assignments[job_id] = tuple(worker_ids_for_job)
        num_workers_assigned += scale_factor

        for single_job_id in job_id.singletons():
            self._per_job_latest_timestamps[single_job_id] = \
                self.get_current_timestamp()
            self._running_jobs.add(single_job_id)

        # Update state.
        worker_state['worker_ids'] = worker_ids
        worker_state['server_id_ptr'] = server_id_ptr
        worker_state['num_workers_assigned'] = num_workers_assigned

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper(self, worker_types):
        """Greedily selects the jobs to run in the next round by iterating
           through the job list in sorted priority order.

           Args:
             worker_types: An ordered list of worker types.

           Returns:
             A list of job IDs and associated scale factors to schedule for the
             upcoming round.
        """

        already_scheduled_jobs = set()
        scheduled_jobs = {}

        num_workers_left = {}
        for worker_type in worker_types:
            scheduled_jobs[worker_type] = []
            num_workers = self._cluster_spec[worker_type]
            num_workers_left[worker_type] = num_workers

        sorted_job_queue = []
        for worker_type in worker_types:
            per_worker_type_entries = []
            for job_id in self._priorities[worker_type]:
                allocation = 0.0
                if self._allocation is not None and job_id in self._allocation:
                    allocation = self._allocation[job_id][worker_type]
                per_worker_type_entries.append(
                        (job_id, worker_type,
                         self._priorities[worker_type][job_id],
                         self._deficits[worker_type][job_id],
                         allocation))
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

        new_worker_assignments = collections.OrderedDict()
        scheduled_jobs = self._schedule_jobs_on_workers_helper(worker_types)

        worker_state = {}
        for worker_type in worker_types:
            # Sort jobs by the scale factor: want to assign jobs from largest
            # to smallest to minimize fragmentation.
            scheduled_jobs[worker_type].sort(key=lambda x: x[1], reverse=True)
            worker_ids = copy.copy(
                self._worker_type_to_worker_id_mapping[worker_type])
            worker_state[worker_type] = {
                'worker_ids': worker_ids,
                'reserved_worker_ids': set(),
                'server_id_ptr': 0,
                'num_workers_assigned': 0,
            }

        # Keep jobs on the same server if possible.
        already_scheduled_jobs = set()
        new_worker_types = {}
        for worker_type in scheduled_jobs:
            for (job_id, scale_factor) in scheduled_jobs[worker_type]:
                new_worker_types[job_id] = worker_type
        for (job_id, worker_ids) in self._current_worker_assignments.items():
            current_worker_type = \
                self._worker_id_to_worker_type_mapping[worker_ids[0]]
            if job_id not in new_worker_types:
                continue
            if new_worker_types[job_id] == current_worker_type:
                reserved_worker_ids = \
                    worker_state[current_worker_type]['reserved_worker_ids']
                scale_factor = len(worker_ids)
                for worker_id in worker_ids:
                    reserved_worker_ids.add(worker_id)
                already_scheduled_jobs.add(job_id)
                self._assign_workers_to_job(job_id, scale_factor,
                                            current_worker_type,
                                            worker_state[current_worker_type],
                                            new_worker_assignments,
                                            worker_ids)

        # Assign all jobs that have an updated placement.
        for worker_type in worker_types:
            for (job_id, scale_factor) in scheduled_jobs[worker_type]:
                if self._allocation is None or job_id not in self._allocation:
                    continue
                elif job_id in already_scheduled_jobs:
                    continue
                self._assign_workers_to_job(job_id, scale_factor, worker_type,
                                            worker_state[worker_type],
                                            new_worker_assignments)
        return new_worker_assignments

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
            scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
            job_types = []
            for single_job_id in single_job_ids:
                job_types.append((self._jobs[single_job_id].job_type,
                                  scale_factor))
            oracle_throughput =\
                oracle_throughputs[job_types[0]][job_types[1]]
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


    def _save_checkpoint(self, checkpoint_file,
                         last_job_arrival_time,
                         next_job_arrival_time,
                         current_round_start_time,
                         current_round_end_time,
                         running_jobs):
        with open(checkpoint_file, 'wb') as f:
            import pickle
            pickle.dump(self._completed_jobs, f)
            pickle.dump(last_job_arrival_time, f)
            pickle.dump(next_job_arrival_time, f)
            pickle.dump(current_round_start_time, f)
            pickle.dump(current_round_end_time, f)
            pickle.dump(running_jobs, f)

            pickle.dump(self._jobs, f)
            pickle.dump(self._throughputs, f)
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
            self._completed_jobs = pickle.load(f)
            last_job_arrival_time = pickle.load(f)
            next_job_arrival_time = pickle.load(f)
            current_round_start_time = pickle.load(f)
            current_round_end_time = pickle.load(f)
            running_jobs = pickle.load(f)

            self._jobs = pickle.load(f)
            self._throughputs = pickle.load(f)
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

            return (last_job_arrival_time,
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

        # Sample the job duration and scale factor from the Philly job
        # distribution.
        if fixed_job_duration:
            print('Running for fixed duration '
                  '%d minutes' % (fixed_job_duration / 60.0))
            run_time = fixed_job_duration
            # TODO: Select the scale factor from the distribution here?
            scale_factor = 1
        else:
            scale_factor = 1
            r = self._job_generator.uniform(0, 1)
            if 0.7 <= r <= 0.8:
                scale_factor = 2
            elif 0.8 <= r <= 0.95:
                scale_factor = 4
            elif 0.95 <= r:
                scale_factor = 8
            if self._job_generator.random() >= 0.8:
                run_time = 60 * (10 ** self._job_generator.uniform(3, 4))
            else:
                run_time = 60 * (10 ** self._job_generator.uniform(1.5, 3))
        if not generate_multi_gpu_jobs:
            scale_factor = 1
        assert(run_time > 0)
        assert(scale_factor >= 1 and scale_factor <= 8)

        # Sample the job type.
        while True:
            job_template = self._job_generator.choice(JobTable)
            if (scale_factor == 1 or
                (scale_factor > 1 and job_template.distributed)):
                break
        job_type = job_template.model

        # Complete the job command with the run directory.
        if job_template.needs_data_dir:
            command = job_template.command % (run_dir, run_dir)
        else:
            command = job_template.command % (run_dir)

        # Compute the number of steps the job will run for given its duration.
        key = (job_type, scale_factor)
        assert(key in self._oracle_throughputs['v100'])
        num_steps = \
            (run_time *
             self._oracle_throughputs['v100'][key]['null'])
        assert(num_steps > 0)

        # Optionally assign a priority to the job.
        priority_weight = 1.0
        if generate_multi_priority_jobs:
            r = self._job_generator.uniform(0, 1)
            if 0.0 <= r <= 0.2:
                priority_weight = 5.0

        # Optionally assign an SLO to the job.
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
                 num_gpus_per_server=None,
                 ideal=False,
                 output_trace_file_name=None):
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

        if not from_trace and output_trace_file_name is not None:
            output_trace_file = open(output_trace_file_name, 'w')
        else:
            output_trace_file = None

        running_jobs = []
        num_jobs_generated = 0
        last_job_arrival_time = None
        next_job_arrival_time = 0
        if arrival_times is not None and len(arrival_times) > 0:
            next_job_arrival_time = arrival_times[0]
        no_dispatched_or_running_jobs = False
        current_round_start_time = 0
        current_round_end_time = None
        window_start_time = None

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
            (last_job_arrival_time,
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
                num_remaining_workers = cluster_spec[worker_type]
                while num_remaining_workers > 0:
                    job = self._generate_job(
                        fixed_job_duration=fixed_job_duration,
                        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                        generate_multi_priority_jobs=generate_multi_priority_jobs)
                    if ((jobs_to_complete is None or
                         window_start_time is not None) and
                        output_trace_file is not None):
                        output_trace_file.write('%s\t%f\n' % (str(job), 0))
                    num_remaining_workers -= job.scale_factor
                    num_jobs_generated += 1
                    self._all_jobs.append((0, job))
                    job_id = self.add_job(job, timestamp=0)

        while True:
            if debug:
                input('Press Enter to continue...')
            if jobs_to_complete is not None:
                num_completed_jobs = \
                    len(jobs_to_complete.intersection(self._completed_jobs))
                print('Number of completed jobs: %d' % (num_completed_jobs))
                if self.is_done(jobs_to_complete):
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
                    self._in_progress_updates[job_id] = []
                    scale_factor =\
                        self._jobs[job_id.singletons()[0]].scale_factor
                    total_steps = [0] * len(job_id.singletons())
                    for i, worker_id in enumerate(worker_ids):
                        if i == len(worker_ids) - 1:
                            # For the last worker, assign all remaining
                            # steps to account for any rounding error.
                            all_num_steps_ = []
                            for j in range(len(all_num_steps)):
                                remaining_steps = \
                                    all_num_steps[j] - total_steps[j]
                                all_num_steps_.append(remaining_steps)
                        else:
                            # Each worker gets an equal fraction of the total
                            # number of steps.
                            all_num_steps_ = \
                                [x // scale_factor for x in all_num_steps]
                        for j in range(len(all_num_steps_)):
                            total_steps[j] += all_num_steps_[j]
                        self._done_callback(job_id, worker_id,
                                            all_num_steps_,
                                            all_execution_times)
                    for single_job_id in job_id.singletons():
                        if single_job_id not in self._jobs:
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
                        if (jobs_to_complete is not None and
                            job_id == min(jobs_to_complete)):
                            window_start_time = self._current_timestamp
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
                    if (jobs_to_complete is not None and
                        job_id == min(jobs_to_complete)):
                        window_start_time = next_job_arrival_time
                        if output_trace_file is not None:
                            print('%d running jobs '
                                  'at window start' % (len(self._jobs) - 1))
                            # Dump already running jobs.
                            for running_job_id in sorted(self._jobs.keys()):
                                remaining_steps = \
                                    self._get_remaining_steps(running_job_id)
                                total_steps = \
                                    self._jobs[running_job_id].total_steps
                                self._jobs[running_job_id]._total_steps = \
                                    remaining_steps
                                output_trace_file.write(
                                    '%s\t0\n' % (str(self._jobs[running_job_id])))
                                self._jobs[running_job_id]._total_steps = \
                                    total_steps
                    if ((jobs_to_complete is None or
                         window_start_time is not None) and
                        output_trace_file is not None):
                        output_arrival_time = next_job_arrival_time
                        if window_start_time is not None:
                            output_arrival_time -= window_start_time
                        output_trace_file.write('%s\t%f\n' % (str(job),
                                                 output_arrival_time))
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
            if ideal:
                time_to_next_event = next_job_arrival_time - self._current_timestamp
                all_num_steps = {}
                self._update_priorities()
                if self._allocation is None:
                    continue
                for job_id in self._allocation:
                    for worker_type in self._allocation[job_id]:
                        time_spent_on_worker_type = self._allocation[job_id][worker_type] * \
                            time_to_next_event
                        if job_id.is_pair():
                            for i, single_job_id in enumerate(job_id.singletons()):
                                if job_id not in self._throughputs:
                                    continue
                                num_steps = time_spent_on_worker_type * \
                                    self._throughputs[job_id][worker_type][i]
                                if single_job_id not in all_num_steps:
                                    all_num_steps[single_job_id] = 0
                                all_num_steps[single_job_id] += int(num_steps)
                        else:
                            if job_id in self._throughputs:
                                num_steps = time_spent_on_worker_type * \
                                    self._throughputs[job_id][worker_type]
                                if job_id not in all_num_steps:
                                    all_num_steps[job_id] = 0
                                all_num_steps[job_id] += int(num_steps)
                for job_id in all_num_steps:
                    allocation_str = ''
                    for x in worker_types:
                        allocation_str += \
                            ' [%4s %f]' % (x, self._allocation[job_id][x])
                    print(('%s]\t[Micro-task scheduled]\tJob ID: %s\t'
                           'Allocation: %s') % (self.get_current_timestamp(),
                                                job_id,
                                                allocation_str))
                    heapq.heappush(running_jobs, (-next_job_arrival_time, job_id,
                                                  (0,),
                                                  [all_num_steps[job_id]]))
                    self._running_jobs.add(job_id)
            else:
                with self._scheduler_lock:
                    scheduled_jobs = self._schedule_jobs_on_workers()
                    self._current_worker_assignments = scheduled_jobs
                    self._print_schedule_summary()
                for (job_id, worker_ids) in scheduled_jobs.items():
                    worker_type = \
                        self._worker_id_to_worker_type_mapping[worker_ids[0]]
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
                                      last_job_arrival_time,
                                      next_job_arrival_time,
                                      current_round_start_time,
                                      current_round_end_time,
                                      running_jobs)
                checkpoint_complete = True

        if window_start_time is not None:
            print('Window start time: %f' % (window_start_time))
            window_duration = self._current_timestamp - window_start_time
            print('Window duration: '
                  '%.3f seconds (%.2f hours)' % (window_duration,
                                                 window_duration / 3600.0))
        if output_trace_file is not None:
            output_trace_file.close()
        print('Total duration: %.3f seconds '
              '(%.2f hours)' % (self._current_timestamp,
                                self._current_timestamp / 3600.0))

    def _recompute_schedule_and_extend_leases(self):
        # Recompute the schedule for the upcoming round.
        self._next_worker_assignments = self._schedule_jobs_on_workers()
        # Check whether we should update the lease for any jobs.
        for job_id in self._current_worker_assignments:
            current_worker_ids = \
                set(self._current_worker_assignments[job_id])
            if job_id in self._next_worker_assignments:
                next_worker_ids = \
                    set(self._next_worker_assignments[job_id])
                if current_worker_ids == next_worker_ids:
                    # Job will be scheduled on the same workers in
                    # upcoming round; extend its lease.
                    self._write_queue.put(
                        'Extending lease of job %s' % (job_id))
                    self._jobs_with_extended_lease.add(job_id)
                elif job_id in self._jobs_with_extended_lease:
                    # Job will not be scheduled on the same workers
                    # in upcoming round; remove it from the
                    # extended lease set if it had previously
                    # received an extended lease.
                    self._jobs_with_extended_lease.remove(job_id)
            elif job_id in self._jobs_with_extended_lease:
                # Job will not be scheduled in upcoming round;
                # remove it from the extended lease set if it
                # had previously received an extended lease.
                self._jobs_with_extended_lease.remove(job_id)

    def _try_dispatch_job(self, job_id, worker_ids):
        """Attempts to dispatch the specified job combination.

           Updates relevant metadata and returns if job has already been
           dispatched.
        """
        # Job could have been completed after schedule was
        # computed; if so, update port data if necessary and return.
        if job_id not in self._jobs:
            scale_factor = len(worker_ids)
            if scale_factor > 1:
                master_addr = \
                    self._worker_connections[worker_ids[0]].addr
                if master_addr not in master_port_offsets:
                    self._master_port_offsets[master_addr] = 1
                self._master_port_offsets[master_addr] += \
                    len(job_id.singletons())
            return
        master_addr = None
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
                    if master_addr not in self._master_port_offsets:
                        self._master_port_offsets[master_addr] = 1
                    offset = self._master_port_offsets[master_addr]
                    master_port = \
                        self._worker_connections[worker_ids[0]].port
                    command = ('%s --master_addr %s '
                               '--master_port %d '
                               '--world_size %d '
                               '--rank %d' % (command,
                                              master_addr,
                                              master_port + j + offset,
                                              scale_factor, i))
                job_descriptions.append(
                        (single_job_id,
                         command,
                         self._jobs[single_job_id].needs_data_dir,
                         self._jobs[single_job_id].num_steps_arg,
                         num_steps))
            # Do not dispatch the job again if it is already
            # running.
            if job_id not in self._dispatched_jobs:
                self._worker_connections[worker_id].run(
                        job_descriptions, worker_id)
                if i == len(worker_ids) - 1:
                    self._dispatched_jobs.add(job_id)
            self._remove_available_worker_id(worker_id)
        # Reset update metadata.
        self._in_progress_updates[job_id] = []
        self._lease_update_requests[job_id] = []
        self._max_steps[job_id] = None
        if master_addr is not None:
            self._master_port_offsets[master_addr] += \
                len(job_id.singletons())

    def _end_round(self):
        """Ends the current round."""
        # Reset extended leases.
        jobs_with_extended_lease = list(self._jobs_with_extended_lease)
        for job_id in jobs_with_extended_lease:
            if job_id in self._jobs:
                current_worker_ids = \
                    self._current_worker_assignments[job_id]
                for worker_id in current_worker_ids:
                    self._add_available_worker_id(worker_id)
            # NOTE: It is possibile that a job which should receive an extended
            # lease requests a lease update after we have removed the job from
            # the extended lease set but before we begin the next round -
            # in this case the job will simply be re-scheduled on the exact
            # same workers. While this is not optimal, it is safe and also
            # unlikely to occur in practice given a sufficiently large delta
            # between the lease update request time and the end of the round.
            self._jobs_with_extended_lease.remove(job_id)

        assert(self._next_worker_assignments is not None)
        self._current_worker_assignments = self._next_worker_assignments
        self._next_worker_assignments = None

        if len(self._dispatched_jobs) > 0:
            self._num_completed_rounds += 1

    def _schedule_with_rounds_async(self):
        self._scheduler_cv.acquire()
        # Wait for jobs to arrive and all workers to register with scheduler.
        while (len(self._jobs) == 0 or
                (self._expected_num_workers is not None and
                 len(self._worker_ids) < self._expected_num_workers)):
            self._scheduler_cv.wait()

        # Add all workers to the queue.
        for worker_id in self._worker_ids:
            self._available_worker_ids.put(worker_id)

        # Wait for allocation to be computed.
        while self._allocation == {}:
            self._scheduler_cv.wait()

        # Compute initial schedule.
        self._current_worker_assignments = self._schedule_jobs_on_workers()
        self._print_schedule_summary()
        self._scheduler_cv.release()

        while True:
            round_start_time = self.get_current_timestamp(in_seconds=True)
            recompute_schedule_time = round_start_time + \
                    (self._time_per_iteration * SCHEDULE_RECOMPUTE_FRACTION)
            round_end_time = round_start_time + self._time_per_iteration

            # Dispatch jobs.
            with self._scheduler_lock:
                for (job_id, worker_ids) in \
                    self._current_worker_assignments.items():
                    self._try_dispatch_job(job_id, worker_ids)

            # Compute the schedule for the upcoming round partway through the
            # current round and extend leases if necessary.
            time.sleep(recompute_schedule_time - round_start_time)
            with self._scheduler_lock:
                self._recompute_schedule_and_extend_leases()
                self._master_port_offsets = {}

            # End the current round.
            current_time = self.get_current_timestamp(in_seconds=True)
            time.sleep(round_end_time - current_time)
            with self._scheduler_lock:
                self._end_round()
            if self.is_done():
                break

    def _schedule_with_rounds(self):
        """Schedules jobs on workers using rounds.

        In a loop, schedules in rounds the applications most in need of
        being run (that is, the applications with the highest
        fraction_allocated/fraction_run ratio) using a DP algorithm.
        """

        recompute_schedule_time = (self._time_per_iteration *
                                   SCHEDULE_RECOMPUTE_FRACTION)
        while True:
            time.sleep(5)
            with self._scheduler_lock:
                num_workers = len(self._worker_ids)
                num_jobs = len(self._jobs)
                if num_workers == 0 or num_jobs == 0:
                    continue
                elif (self._expected_num_workers is not None and
                      num_workers < self._expected_num_workers):
                    # Wait for all workers to be launched before starting
                    # to dispatch jobs.
                    # TODO: Replace this with cluster_spec?
                    continue
                # Reset available_worker_ids to the desired size.
                self._available_worker_ids = set_queue.Queue(num_workers)
                for worker_id in self._worker_ids:
                    self._add_available_worker_id(worker_id)
                if self._next_worker_assignments is not None:
                    scheduled_jobs = self._next_worker_assignments
                    self._next_worker_assignments = None
                else:
                    scheduled_jobs = self._schedule_jobs_on_workers()
                self._current_worker_assignments = scheduled_jobs
                self._print_schedule_summary()
                self._master_port_offsets = {}
                assert(len(self._jobs_with_extended_lease) == 0)
                for (job_id, worker_ids) in scheduled_jobs.items():
                    self._try_dispatch_job(job_id, worker_ids)
            round_start_time = self.get_current_timestamp(in_seconds=True)
            while not self._available_worker_ids.full():
                with self._scheduler_lock:
                    current_time = self.get_current_timestamp(in_seconds=True)
                    elapsed_time = current_time - round_start_time
                    if (elapsed_time >= recompute_schedule_time and
                        self._next_worker_assignments is None):
                        # If the specified duration of the current round has
                        # completed, compute the schedule for the upcoming
                        # round.
                        self._recompute_schedule_and_extend_leases()
                    elif elapsed_time >= self._time_per_iteration:
                        # When the round completes, reset any extended leases.
                        jobs_with_extended_lease = \
                            list(self._jobs_with_extended_lease)
                        for job_id in jobs_with_extended_lease:
                            if job_id in self._jobs:
                                current_worker_ids = \
                                    self._current_worker_assignments[job_id]
                                for worker_id in current_worker_ids:
                                    self._add_available_worker_id(worker_id)
                            # NOTE: There is a possibility that a job which
                            # should receive an extended lease requests a lease
                            # update after we have removed the job from the
                            # extended lease set but before we begin the
                            # next round - in this case the job will
                            # simply be re-scheduled on the exact same workers.
                            # While this is not optimal, it is safe and also
                            # unlikely to occur in practice.
                            self._jobs_with_extended_lease.remove(job_id)
                time.sleep(2)
                continue
            # Only record a completed round if at least one job was active.
            if len(self._dispatched_jobs) > 0:
                self._num_completed_rounds += 1
            if self.is_done():
                break

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
            else:
                job_ids = sorted(job_ids)
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
                print('Average job completion time: %.3f seconds '
                      '(%.2f hours)' % (average_job_completion_time,
                                        average_job_completion_time / 3600.0))
                if len(low_priority_job_completion_times) > 0:
                    average_low_pri_jct = \
                        np.mean(low_priority_job_completion_times)
                    print('Average job completion time (low priority): '
                          '%.3f seconds '
                          '(%.2f hours)' % (average_low_pri_jct,
                                            average_low_pri_jct / 3600.0))
                if len(high_priority_job_completion_times) > 0:
                    average_high_pri_jct = \
                        np.mean(high_priority_job_completion_times)
                    print('Average job completion time (high priority): '
                          '%.3f seconds '
                          '(%.2f hours)' % (average_high_pri_jct,
                                            average_high_pri_jct / 3600.0))
            return average_job_completion_time


    def get_completed_steps(self, job_ids=None):
        print('Completed steps:')
        if job_ids is None:
            job_ids = sorted(list(self._total_steps_run.keys()))
        else:
            job_ids = sorted(job_ids)
        for job_id in job_ids:
            if job_id in self._total_steps_run:
                completed_steps = self._total_steps_run[job_id]
                print('Job %s: %d steps' % (job_id, completed_steps))

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

    def _get_allocation_state(self):
        """Prepare all relevant scheduler state for computing the allocation."""
        state = {}
        state['scale_factors'] = {
            job_id: self._jobs[job_id].scale_factor
            for job_id in self._jobs
        }
        state['priority_weights'] = {
            job_id: self._jobs[job_id].priority_weight
            for job_id in self._jobs
        }
        state['num_steps_remaining'] = {
            job_id: self._get_remaining_steps(job_id)
            for job_id in self._jobs
        }
        state['times_since_start'] = {
            job_id: self.get_current_timestamp() - \
                        self._per_job_start_timestamps[job_id]
            for job_id in self._jobs
        }
        state['throughputs'] = copy.deepcopy(self._throughputs)
        state['cluster_spec'] = copy.deepcopy(self._cluster_spec)

        if self._policy.name.startswith("ThroughputNormalizedByCostSum"):
            state['instance_costs'] = copy.deepcopy(self._per_worker_type_prices)
            if 'SLO' in self._policy.name:
                SLOs = {}
                if self._SLOs is not None:
                    for job_id in self._jobs:
                        SLOs[job_id] = \
                            (self._SLOs[job_id] -
                             self.get_current_timestamp(in_seconds=True))
                    state['SLOs'] = SLOs
                else:
                    state['num_steps_remaining'] = {}
        return state

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _compute_allocation(self, state=None):
        """Computes the allocation.

        Uses the specified policy to compute an allocation of jobs to
        compute resources. Requires self._scheduler_lock to be held
        when calling this function.

        Returns:
            A 2-level dict indexed by job_id and then worker_type (the
            unflattened allocation). For example,

            {0: {"v100": 0.25, "p100": 0.95}, 1: {"v100": 0.75, "p100": 0.05}}

            indicates that for 25% of the time, worker type 'v100' should run,
            job 0 and for 95% of the time, worker type 'p100' should run job 0.
        """
        if state is None:
            state = self._get_allocation_state()
        throughputs = state['throughputs']
        scale_factors = state['scale_factors']
        times_since_start = state['times_since_start']
        num_steps_remaining = state['num_steps_remaining']
        priority_weights = state['priority_weights']
        cluster_spec = state['cluster_spec']

        # Compute the allocation.
        if self._policy.name == "AlloX_Perf":
            allocation = self._policy.get_allocation(
                throughputs, scale_factors,
                times_since_start, num_steps_remaining,
                cluster_spec)
        elif self._policy.name.startswith("FinishTimeFairness"):
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, priority_weights,
                times_since_start, num_steps_remaining,
                cluster_spec)
        elif self._policy.name == "Isolated":
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, cluster_spec)
        elif self._policy.name.startswith("MaxMinFairness"):
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, priority_weights,
                cluster_spec)
        elif self._policy.name.startswith("MinTotalDuration"):
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, num_steps_remaining,
                cluster_spec)
        elif self._policy.name.startswith("ThroughputNormalizedByCostSum"):
            instance_costs = state['instance_costs']
            if 'SLO' in self._policy.name:
                SLOs = state['SLOs']
                allocation = self._policy.get_allocation(
                    throughputs, scale_factors, cluster_spec,
                    instance_costs=instance_costs,
                    SLOs=SLOs,
                    num_steps_remaining=num_steps_remaining)
            else:
                allocation = self._policy.get_allocation(
                    throughputs, scale_factors, self._cluster_spec,
                    instance_costs=instance_costs)
        else:
            allocation = self._policy.get_allocation(
                throughputs, scale_factors, self._cluster_spec)
        if allocation is None:
            allocation = {}
        return allocation

    def _allocation_thread(self):
        """Computes the allocation asynchronously."""
        while True:
            # Check whether allocation needs to be re-computed.
            self._scheduler_cv.acquire()
            while not self._need_to_update_allocation:
                self._scheduler_cv.wait()
            state = self._get_allocation_state()
            self._scheduler_cv.release()
            allocation = self._compute_allocation(state)

            # Update allocation and clean up.
            self._scheduler_cv.acquire()
            for job_id in allocation:
                still_active = []
                for single_job_id in job_id.singletons():
                    if single_job_id in self._jobs:
                        still_active.append(True)
                    else:
                        still_active.append(False)
                if not all(still_active):
                    worker_types = allocation[job_id].keys()
                    for i, single_job_id in enumerate(job_id.singletons()):
                        if still_active[i]:
                            # If only one job in a job combination is still
                            # active, re-distribute the job combination's
                            # allocation to the still-active job's isolated
                            # allocation.
                            for worker_type in worker_types:
                                allocation[single_job_id][worker_type] += \
                                    allocation[job_id][worker_type]
                                del allocation[job_id][worker_type]
                            del allocation[job_id]
            self._allocation = allocation
            self._need_to_update_allocation = False
            self._scheduler_cv.notify()
            self._scheduler_cv.release()

    # @preconditions(lambda self: self._simulate or self._scheduler_lock.locked())
    def _populate_job_combination_metadata(self, job_id, worker_type):
        """Populate metadata for job combinations involving passed-in job_id."""

        job = self._jobs[job_id]
        job_type_key = (job.job_type, job.scale_factor)
        if self._estimate_throughputs:
            assert(job.scale_factor == 1)
            reference_throughputs = self._reference_throughputs[worker_type]
        for other_job_id in self._jobs:
            if other_job_id != job_id:
                other_job = self._jobs[other_job_id]
                if job.scale_factor != other_job.scale_factor:
                    continue
                other_job_type_key = (other_job.job_type, other_job.scale_factor)
                job_type_keys = [job_type_key, other_job_type_key]
                merged_job_id = \
                        job_id_pair.JobIdPair(job_id[0], other_job_id[0])
                if merged_job_id not in self._throughputs:
                    self._throughputs[merged_job_id] = {}
                    self._job_time_so_far[merged_job_id] = {}
                    self._priorities[worker_type][job_id] = 0.0
                    self._deficits[worker_type][job_id] = 0.0
                self._job_time_so_far[merged_job_id][worker_type] = 0.0
                if self._estimate_throughputs:
                    reference_job_types = \
                        [self._reference_job_map[job_id],
                         self._reference_job_map[other_job_id]]
                    isolated_throughputs = \
                        [self._oracle_throughputs[worker_type][job_type_key]['null'],
                         self._oracle_throughputs[worker_type][other_job_type_key]['null']]
                    if job_id < other_job_id:
                        self._throughputs[merged_job_id][worker_type] = \
                            np.multiply(
                                reference_throughputs[reference_job_types[0]][reference_job_types[1]],
                                isolated_throughputs)
                    else:
                        self._throughputs[merged_job_id][worker_type] = \
                            np.multiply(
                                reference_throughputs[reference_job_types[1]][reference_job_types[0]],
                                isolated_throughputs[::-1])
                elif (self._oracle_throughputs is None or
                    job.scale_factor != other_job.scale_factor):
                    self._throughputs[merged_job_id][worker_type] = [0.0, 0.0]
                else:
                    oracle_throughputs = self._oracle_throughputs[worker_type]
                    # The single-job IDs for job pairs are stored in sorted
                    # order so make sure the co-located throughputs match this
                    # order.
                    scale_factor = job.scale_factor
                    if job_id < other_job_id:
                        self._throughputs[merged_job_id][worker_type] = \
                            oracle_throughputs[job_type_keys[0]][job_type_keys[1]]
                    else:
                        self._throughputs[merged_job_id][worker_type] = \
                            oracle_throughputs[job_type_keys[1]][job_type_keys[0]]

    def _set_initial_throughput(self, job_id, worker_type):
        assert(not job_id.is_pair())
        if self._oracle_throughputs is not None:
            job_type = self._jobs[job_id].job_type
            scale_factor = self._jobs[job_id].scale_factor
            key = (job_type, scale_factor)
            self._throughputs[job_id][worker_type] = \
                self._oracle_throughputs[worker_type][key]['null']
        else:
            self._throughputs[job_id][worker_type] = DEFAULT_THROUGHPUT

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
        time_since_last_reset = self.get_current_timestamp() - self._last_reset_time
        reset_interval_elapsed = time_since_last_reset >= \
            self._minimum_time_between_allocation_resets
        if (self._need_to_update_allocation and
            (reset_interval_elapsed or self._last_reset_time == 0)):
            self._reset_time_run_so_far()
            # In simulation mode, wait for allocation computation to complete
            # before proceeding.
            if self._simulate:
                self._allocation = self._compute_allocation()
                self._need_to_update_allocation = False

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            for job_id in self._job_time_so_far:
                if self._worker_time_so_far[worker_type] == 0.0 or worker_type not in self._job_time_so_far[job_id]:
                    fraction = 0.0
                else:
                    fraction = self._job_time_so_far[job_id][worker_type] / \
                             self._worker_time_so_far[worker_type]
                fractions[worker_type][job_id] = fraction
            for job_id in self._priorities[worker_type]:
                # Don't use inf so 2*new_priority > new_priority.
                #
                # Scale the default value by the allocation so that newly
                # added jobs run according to their respective allocations.
                if self._allocation is None or job_id not in self._allocation:
                    self._priorities[worker_type][job_id] = 0.0
                else:
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
                return self._available_worker_ids.get_nowait(item=worker_id)
            except queue.Empty as e:
                return None
        else:
            return self._available_worker_ids.get(item=worker_id)

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
            self._all_rpc_clients.append(rpc_client)

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
            self._scheduler_cv.notify()

        return (per_worker_ids, self._time_per_iteration)

    def _update_lease_callback(self, job_id, worker_id, steps, duration,
                               max_steps, max_duration):
        # Round the remaining steps to the nearest multiple of scale_factor.
        scale_factor = self._jobs[job_id].scale_factor
        remaining_steps = self._get_remaining_steps(job_id)
        remaining_steps = int(math.ceil(remaining_steps / scale_factor))

        if steps == 0 or duration == 0:
            return (remaining_steps, self._time_per_iteration)

        # Extend the lease if the job has been placed on the same workers
        # for the upcoming round.
        with self._scheduler_lock:
            if job_id in self._jobs_with_extended_lease:
                return (max_steps, max_duration + self._time_per_iteration)

        if scale_factor == 1:
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
                    self._max_steps[job_id] = \
                        min(remaining_steps,
                            steps + int(remaining_time * throughput))
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

        to_remove = []
        with self._scheduler_lock:
            current_timestamp = self.get_current_timestamp()
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            self._add_available_worker_id(worker_id)

            scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
            self._in_progress_updates[job_id].append((worker_id,
                                                      all_num_steps,
                                                      all_execution_times))
            if len(self._in_progress_updates[job_id]) < scale_factor:
                return
            else:
                assert(job_id in self._dispatched_jobs)
                self._dispatched_jobs.remove(job_id)
                micro_task_succeeded = True
                all_worker_ids = \
                    [x[0] for x in self._in_progress_updates[job_id]]
                all_worker_ids.sort()
                all_num_steps = [0] * len(job_id.singletons())
                all_execution_times = [0] * len(job_id.singletons())
                for (_, all_num_steps_, all_execution_times_) in \
                    self._in_progress_updates[job_id]:
                    if (np.min(all_num_steps_) <= 0 or
                        np.min(all_execution_times_) <= 0):
                        micro_task_succeeded = False
                        break
                    for i in range(len(job_id.singletons())):
                        all_num_steps[i] += all_num_steps_[i]
                        all_execution_times[i] = max(all_execution_times[i],
                                                     all_execution_times_[i])

            if not micro_task_succeeded:
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
                self._scheduler_cv.notify()

            else:
                print(('%s]\t[Micro-task succeeded]\t'
                       'Job ID: %s\tWorker type: %s\t'
                       'Worker ID(s): %s') % (current_timestamp,
                                           job_id,
                                           worker_type,
                                           str(all_worker_ids)))
                self._num_failures_per_job[job_id] = 0
                for single_job_id, num_steps, execution_time in \
                        zip(job_id.singletons(), all_num_steps,
                            all_execution_times):
                    if self._per_worker_type_prices is not None:
                        self._job_cost_so_far[single_job_id] += \
                            (self._per_worker_type_prices[worker_type] *
                             execution_time / 3600.0 * scale_factor)
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
                if job_id in self._job_time_so_far:
                    self._job_time_so_far[job_id][worker_type] += \
                        max_execution_time
                    self._worker_time_so_far[worker_type] += \
                        max_execution_time
                for worker_id in all_worker_ids:
                    self._cumulative_worker_time_so_far[worker_id] += \
                        max_execution_time

        self._update_throughput(job_id, worker_type,
                                all_num_steps,
                                all_execution_times)

        for single_job_id in to_remove:
            self.remove_job(single_job_id[0])

        # Try to dispatch the next job that uses this worker.
        dispatch_next_job = False
        if not self._simulate and self._next_worker_assignments is not None:
            # Check whether this worker has been assigned a new job.
            for next_job_id in self._next_worker_assignments:
                for next_worker_id in self._next_worker_assignments[job_id]:
                    if worker_id == next_worker_id:
                        dispatch_next_job = True
                        next_worker_ids = self._next_worker_assignments[job_id]
                        break
        if dispatch_next_job:
            # Ensure that all workers used by the next job are available.
            for next_worker_id in next_worker_ids:
                if next_worker_id not in self._available_worker_ids:
                    return
            self._write_queue.put(
                'Trying to dispatch job %s early' % (next_job_id))
            self._try_dispatch_job(next_job_id, next_worker_ids)
