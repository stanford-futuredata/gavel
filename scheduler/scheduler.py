from __future__ import print_function

import collections
import copy
import faulthandler
import heapq
import numpy as np
import os
# from preconditions import preconditions
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import datetime
import random
import sched
import math
import matrix_completion
import warnings
import logging

# TODO: clean these up.
from job import Job
import job_id_pair
from job_table import JobTable
from runtime.rpc import scheduler_server, scheduler_client
import set_queue
from custom_logging import SchedulerAdapter
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
# Format string for logging.
LOG_FORMAT = '{name}:{levelname} {message}'
# Buffer time for jobs to complete.
JOB_COMPLETION_BUFFER_TIME = 60
# Base port to use for distributed jobs.
BASE_JOB_PORT = 60570
# Maximum port number.
MAX_PORT = 65535

class Scheduler:

    # TODO: Make assign_SLOs a configurable parameter from scripts.
    def __init__(self, policy, simulate=False, throughputs_file=None,
                 seed=0, time_per_iteration=360, profiling_percentage=1.0,
                 num_reference_models=len(JobTable),
                 per_instance_type_prices_dir=None,
                 available_clouds=[],
                 assign_SLOs=False,
                 enable_global_queue=False,
                 expected_num_workers=None,
                 minimum_time_between_allocation_resets=1920,
                 max_rounds=None):

        # Flag to control whether scheduler runs in simulation mode.
        self._simulate = simulate

        # Initial timestamp.
        if self._simulate:
            self._start_timestamp = 0
        else:
            self._start_timestamp = time.time()
        # Latest simulated timestamp.
        self._current_timestamp = self._start_timestamp

        # Configure logger.
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._orig_logger = logger
        self._logger = \
            SchedulerAdapter(logger,
                             {'scheduler': self,
                              'start_timestamp': datetime.datetime.now()})
        self._logging_handler = ch

        # Print config information.
        if simulate:
            loc = 'in simulation'
        else:
            loc = 'at {addr}:{port}'.format(addr=utils.get_ip_address(),
                                            port=SCHEDULER_PORT)
        self._logger.info(
            'Running scheduler {loc} with the following args: '
            'policy={policy}, seed={seed}, '
            'time_per_iteration={time_per_iteration}, '
            'profiling_percentage={profiling_percentage}, '
            'num_reference_models={num_reference_models}'.format(
                loc=loc, policy=policy.name, seed=seed,
                time_per_iteration=time_per_iteration,
                profiling_percentage=profiling_percentage,
                num_reference_models=num_reference_models))

        # Initialize seeds.
        self._initialize_seeds(seed)
        # Initialize time in seconds each iteration should run for.
        self._time_per_iteration = time_per_iteration

        # Sets whether to use a global queue across all worker types.
        self._enable_global_queue = enable_global_queue

        self._expected_num_workers = expected_num_workers
        self._minimum_time_between_allocation_resets = \
            minimum_time_between_allocation_resets

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
        # Map from job combinations to assigned workers for current round.
        self._current_worker_assignments = collections.OrderedDict()
        # Map from job combinations to assigned workers for the upcoming round.
        self._next_worker_assignments = None
        # Map from job combinations to assigned workers for jobs that need to
        # be re-dispatched on account of finishing early.
        self._redispatched_worker_assignments = collections.OrderedDict()
        # Set of completed jobs in current round.
        self._completed_jobs_in_current_round = set()
        # Set of jobs with an extended lease for the upcoming round.
        self._jobs_with_extended_lease = set()
        # The total number of lease extensions across all jobs.
        self._num_lease_extensions = 0
        # The total number of instances where leasees could have been extended.
        self._num_lease_extension_opportunities = 0
        # Event scheduler to trigger round completions for jobs with
        # extended leases.
        self._completion_event_scheduler = \
            sched.scheduler(time.time, time.sleep)
        # Map from job ID to completion event.
        self._completion_events = {}
        # Map from job ID to timeline of events.
        self._job_timelines = {}
        # Port offset for distributed jobs.
        self._port_offset = 0
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
        # Flag indicating whether allocation has been updated since elapsed
        # time was last reset.
        self._allocation_changed_since_last_time_reset = False
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
        # Indexed by single job IDs.
        self._max_steps = {}
        # All per-round lease update requests for distributed jobs.
        # Indexed by single job IDs.
        self._lease_update_requests = {}
        # List of all RPC clients.
        self._all_rpc_clients = []
        # Currently running jobs.
        self._running_jobs = set()
        # The timestamp when each worker entered the cluster.
        self._worker_start_times = {}
        # Verbose flag.
        self._verbose = False
        # Data structures for debugging.
        self._micro_tasks_per_job = {}
        self._all_jobs = []
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
            'InitJob': self._init_job_callback,
            'UpdateLease': self._update_lease_callback,
            'Done': self._done_callback,
        }

        if not self._simulate:
            faulthandler.enable()
            f = open('.stack_trace.log', 'w')
            faulthandler.dump_traceback_later(30, repeat=True, file=f,
                                              exit=False)

            self._allocation_thread = \
                threading.Thread(target=self._allocation_thread)
            self._allocation_thread.daemon = True
            self._allocation_thread.start()

            self.server_thread = threading.Thread(
                target=scheduler_server.serve,
                args=(port, callbacks))
            self.server_thread.daemon = True
            self.server_thread.start()

            self._mechanism_thread = \
                threading.Thread(target=self._schedule_with_rounds)
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
                self._scheduler_cv.notifyAll()
                self._scheduler_cv.release()

    def _update_throughput(self, job_id, worker_type, all_num_steps,
                           all_execution_times):
        # Job might have already completed.
        if job_id not in self._throughputs:
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
            if job_id.is_pair():
                new_throughput = self._throughputs[job_id][worker_type]
            else:
                new_throughput = [self._throughputs[job_id][worker_type]]
            self._logger.info(
                'Job {job_id} throughput on worker type {worker_type}: '
                '{orig} -> {updated}'.format(
                    job_id=job_id, worker_type=worker_type,
                    orig=str(old_throughput), updated=str(new_throughput)))

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
            self._job_timelines[job_id] = [[] for _ in range(job.scale_factor)]
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
            self._logger.info(
                '[Job dispatched]\tJob ID: {job_id}'.format(job_id=job_id))
            self._scheduler_cv.notifyAll()

        return job_id

    def remove_job(self, job_id):
        """Public-facing interface to _remove_job."""
        with self._scheduler_lock:
            self._remove_job(job_id)
            self._scheduler_cv.notifyAll()

    def _remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        if type(job_id) is int:
            job_id = job_id_pair.JobIdPair(job_id, None)
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
            self._job_completion_times[job_id] = None
        else:
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
                other_job_is_active = \
                    any([x in self._jobs for x in other_job_id.singletons()])
                del self._throughputs[other_job_id]
                del self._job_time_so_far[other_job_id]
                if not other_job_is_active:
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
        # NOTE: Scheduler cv will be notified by calling function.
        self._need_to_update_allocation = True
        self._logger.info('Remaining active jobs: {0}'.format(len(self._jobs)))

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
        if not self._simulate:
            with self._scheduler_lock:
                for rpc_client in self._all_rpc_clients:
                    rpc_client.shutdown()
        self._orig_logger.removeHandler(self._logging_handler)
        self._logging_handler.close()
        # TODO: Any other cleanup?

    """
    ======================================================================
       Scheduler's main schedule() and simulate() methods.
    ======================================================================
    """

    def _get_state_snapshot(self, deepcopy=False):
        if deepcopy:
            state_snapshot = {
                'allocation': copy.deepcopy(self._allocation),
                'priorities': copy.deepcopy(self._priorities),
                'deficits': copy.deepcopy(self._deficits),
            }
        else:
            state_snapshot = {
                'allocation': self._allocation,
                'priorities': self._priorities,
                'deficits': self._deficits,
            }
        return state_snapshot

    def _print_schedule_summary(self, state_snapshot=None):
        if state_snapshot is not None:
            allocation = state_snapshot['allocation']
            priorities = state_snapshot['priorities']
            deficits = state_snapshot['deficits']
        else:
            allocation = self._allocation
            priorities = self._priorities
            deficits = self._deficits

        completed_jobs = set()
        worker_types = sorted(self._cluster_spec.keys())
        for job_id, worker_ids in self._current_worker_assignments.items():
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            if (job_id in self._completed_jobs_in_current_round or
                job_id not in allocation or
                job_id not in priorities[worker_type] or
                job_id not in deficits[worker_type]):
                completed_jobs.add(job_id)

            if not self._simulate and job_id in completed_jobs:
                self._logger.debug('Job {job_id} has already completed on '
                                   '{num_gpus} {worker_type} GPUs'.format(
                                       job_id=job_id, num_gpus=len(worker_ids),
                                       worker_type=worker_type))
                continue
            allocation_str = ''
            for x in worker_types:
                allocation_str += ' [%4s %.2f]' % (x, allocation[job_id][x])
            self._logger.info(
                '[Micro-task scheduled]\tJob ID: {job_id}\t'
                'Worker type: {worker_type}\tWorker ID(s): {worker_ids}\t'
                'Priority: {priority:.2f}\tDeficit: {deficit:.2f}\t'
                'Allocation: {allocation}'.format(
                    job_id=job_id, worker_type=worker_type,
                    worker_ids=",".join([str(x) for x in worker_ids]),
                    priority=priorities[worker_type][job_id],
                    deficit=deficits[worker_type][job_id],
                    allocation=allocation_str))
        num_workers_assigned = {}
        for job_id, worker_ids in self._current_worker_assignments.items():
            if not self._simulate and job_id in completed_jobs:
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
                self._logger.warn(
                    '{num_gpus} GPUs of type {worker_type} left unused. '
                    'Number of active jobs: {num_active_jobs}'.format(
                        num_gpus=unused_workers, worker_type=worker_type,
                        num_active_jobs=len(self._jobs)))

    def _assign_workers_to_job(self, job_id, scale_factor, worker_type,
                               worker_state, worker_assignments):
        """Assign workers to jobs.

        Assigns workers in a strided fashion to minimize the number
        of servers used.

        Args:
          job_id: The job (combination) ID to schedule.
          scale_factor: The number of GPUs requested.
          worker_type: The worker type to allocate.
          worker_state: A dict comprised of the following information:
            worker_ids: Worker IDs organized into servers.
            assigned_worker_ids: The set of worker IDs assigned so far.
            server_id_ptr: The server to assign workers from.
          worker_assignments: A map from job_id to assigned worker_ids tuple.
        """
        worker_ids = worker_state['worker_ids']
        assigned_worker_ids = worker_state['assigned_worker_ids']
        server_id_ptr = worker_state['server_id_ptr']

        if job_id in worker_assignments:
            worker_ids_for_job = list(worker_assignments[job_id])
        else:
            worker_ids_for_job = []
        while (len(worker_ids_for_job) < scale_factor and
               server_id_ptr < len(worker_ids)):
            if len(worker_ids[server_id_ptr]) == 0:
                server_id_ptr += 1
                continue
            worker_id_to_assign = worker_ids[server_id_ptr][0]
            if worker_id_to_assign not in assigned_worker_ids:
                worker_ids_for_job.append(worker_id_to_assign)
                assigned_worker_ids.add(worker_id_to_assign)
            worker_ids[server_id_ptr].pop(0)

        if len(worker_ids_for_job) != scale_factor:
            raise RuntimeError(
                'Could not assign workers to job %s!' % (job_id))

        worker_assignments[job_id] = tuple(worker_ids_for_job)
        worker_state['server_id_ptr'] = server_id_ptr

        for single_job_id in job_id.singletons():
            if self._simulate:
                # This will be done on initialization when running on a
                # physical cluster.
                self._per_job_latest_timestamps[single_job_id] = \
                    self.get_current_timestamp()
                self._running_jobs.add(single_job_id)

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
            worker_ids = copy.deepcopy(
                self._worker_type_to_worker_id_mapping[worker_type])
            worker_state[worker_type] = {
                'worker_ids': worker_ids,
                'assigned_worker_ids': set(),
                'server_id_ptr': 0,
            }

        prev_worker_types = {}
        for (job_id, worker_ids) in self._current_worker_assignments.items():
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            prev_worker_types[job_id] = worker_type

        for worker_type in worker_types:
            per_worker_state = worker_state[worker_type]
            assigned_worker_ids = per_worker_state['assigned_worker_ids']
            current_job = 0
            scale_factors = set([x[1] for x in scheduled_jobs[worker_type]])
            scale_factors = sorted(scale_factors, reverse=True)

            # Assign workers in order of decreasing scale factor to prioritize
            # locality for multi-GPU jobs.
            for current_scale_factor in scale_factors:
                # Try to keep jobs on current workers if possible.
                for (job_id, scale_factor) in scheduled_jobs[worker_type]:
                    if scale_factor != current_scale_factor:
                        continue
                    if (job_id in prev_worker_types and
                        prev_worker_types[job_id] == worker_type):
                        prev_worker_ids = \
                            self._current_worker_assignments[job_id]
                        assert(isinstance(prev_worker_ids, tuple))
                        extend_placement = True
                        for prev_worker_id in prev_worker_ids:
                            if prev_worker_id in assigned_worker_ids:
                                extend_placement = False
                                break
                        if extend_placement:
                            new_worker_assignments[job_id] = prev_worker_ids
                            for prev_worker_id in prev_worker_ids:
                                assigned_worker_ids.add(prev_worker_id)

                # Assign workers for remaining jobs.
                for (job_id, scale_factor) in scheduled_jobs[worker_type]:
                    if scale_factor != current_scale_factor:
                        continue
                    elif job_id not in self._allocation:
                        continue
                    self._assign_workers_to_job(job_id, scale_factor,
                                                worker_type,
                                                per_worker_state,
                                                new_worker_assignments)

        # Verify the assignment.
        num_assignments = {}
        for job_id in new_worker_assignments:
            for worker_id in new_worker_assignments[job_id]:
                if worker_id not in num_assignments:
                    num_assignments[worker_id] = 0
                num_assignments[worker_id] += 1
        for worker_id in num_assignments:
            if num_assignments[worker_id] != 1:
                raise RuntimeError(
                    'Worker {0} was assigned {1} times!'.format(
                        worker_id, num_assignments[worker_id]))

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
                    raise RuntimeError('Throughput for job {job_id} on '
                                       'worker type {worker_type}'
                                       'should not be less than 0!'.format(
                                           job_id=single_job_id,
                                           worker_type=worker_type))
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
        SLO_generator = self._SLO_generator if self._SLOs is not None else None

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
                    job = utils.generate_job(
                            throughputs=self._oracle_throughputs,
                            reference_worker_type='v100',
                            rng=self._job_generator,
                            job_id=None,
                            fixed_job_duration=fixed_job_duration,
                            generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                            generate_multi_priority_jobs=generate_multi_priority_jobs,
                            SLO_rng=SLO_generator)
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
                self._logger.info(
                    'Number of completed jobs: {0}'.format(num_completed_jobs))
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
                    # If no jobs are currently running and we are not yet done,
                    # force a reset.
                    if len(running_jobs) == 0:
                        self._last_reset_time = 0

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
                if next_job_arrival_time is not None:
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
                    job = utils.generate_job(
                            throughputs=self._oracle_throughputs,
                            reference_worker_type='v100',
                            rng=self._job_generator,
                            job_id=None,
                            fixed_job_duration=fixed_job_duration,
                            generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                            generate_multi_priority_jobs=generate_multi_priority_jobs,
                            SLO_rng=SLO_generator)
                    num_jobs_generated += 1
                    self._all_jobs.append((next_job_arrival_time, job))
                    job_id = self.add_job(job, timestamp=next_job_arrival_time)
                    if (jobs_to_complete is not None and
                        job_id == min(jobs_to_complete)):
                        window_start_time = next_job_arrival_time
                        if output_trace_file is not None:
                            self._logger.info(
                                '{0} running jobs at window start'.format(
                                    len(self._jobs) - 1))
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
                    self._logger.info(
                        '[Micro-task scheduled]\tJob ID: {job_id}\t'
                           'Allocation: {allocation}'.format(
                               job_id=job_id, allocation=allocation_str))
                    heapq.heappush(running_jobs, (-next_job_arrival_time, job_id,
                                                  (0,),
                                                  [all_num_steps[job_id]]))
                    self._running_jobs.add(job_id)
            else:
                with self._scheduler_lock:
                    scheduled_jobs = self._schedule_jobs_on_workers()
                    for job_id in self._current_worker_assignments:
                        is_active = \
                            any([x in self._jobs for x in job_id.singletons()])
                        if is_active:
                            self._num_lease_extension_opportunities += 1
                    for job_id in scheduled_jobs:
                        if job_id in self._current_worker_assignments:
                            current_worker_ids = \
                                set(self._current_worker_assignments[job_id])
                            next_worker_ids = set(scheduled_jobs[job_id])
                            if current_worker_ids == next_worker_ids:
                                self._num_lease_extensions += 1
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

    def _is_final_round(self):
        return (self._max_rounds is not None and
                self._num_completed_rounds + 1 == self._max_rounds)

    def _begin_round(self, state_snapshot=None):
        """Executes beginning stage of a scheduling round."""

        self._current_round_start_time = self.get_current_timestamp()
        current_round = self._num_completed_rounds

        # Reset lease update requests.
        for job_id in self._current_worker_assignments:
            for single_job_id in job_id.singletons():
                self._lease_update_requests[single_job_id] = []
                self._max_steps[single_job_id] = None

        # Re-dispatch jobs that had extended leases but completed early.
        for job_id in self._redispatched_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active:
                if job_id not in self._current_worker_assignments:
                    raise RuntimeError(
                        'Trying to re-dispatch job {0} but it has not '
                        'been scheduled for round {1}!'.format(
                            job_id, current_round))
                worker_ids = self._redispatched_worker_assignments[job_id]
                self._logger.info('Re-dispatching job {0} as it completed '
                                  'early but had an extended lease'.format(
                                    job_id))
                self._try_dispatch_job(job_id, worker_ids)
                self._logger.debug('Re-dispatched job {0}'.format(job_id))
        self._redispatched_worker_assignments = collections.OrderedDict()

        self._logger.debug('Finished re-dispatching jobs')

        self._logger.info('*** START ROUND {0} ***'.format(current_round))
        self._print_schedule_summary(state_snapshot)

    def _mid_round(self, pool):
        """Executes intermediate stage of a scheduling round.

        Computes the schedule for the upcoming round partway through the
        current round and extends leases if necessary. Then dispatches jobs
        for the upcoming round and schedules callbacks for jobs with extended
        leases.

        Note that this updates self._next_worker_assignments. We update
        self._current_worker_assignments when we end the round.
        """

        if self._is_final_round():
            self._logger.debug('In final round, not dispatching any more jobs')
            self._jobs_with_extended_leases = set()
            return

        round_end_time = \
            self._current_round_start_time + self._time_per_iteration

        # Recompute the schedule for the upcoming round.
        self._next_worker_assignments = self._schedule_jobs_on_workers()

        # Count how many jobs could be eligible for lease extensions.
        for job_id in self._current_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active:
                self._num_lease_extension_opportunities += 1

        # Check whether we should update the lease for any jobs.
        for job_id in self._current_worker_assignments:
            current_worker_ids = \
                set(self._current_worker_assignments[job_id])
            if (job_id in self._next_worker_assignments and
                job_id not in self._completed_jobs_in_current_round):
                next_worker_ids = \
                    set(self._next_worker_assignments[job_id])
                if current_worker_ids == next_worker_ids:
                    # Job will be scheduled on the same workers in
                    # upcoming round; extend its lease.
                    self._jobs_with_extended_lease.add(job_id)
                    self._logger.info(
                        'Extending lease for job {0}'.format(job_id))
                    self._num_lease_extensions += 1
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

        # Dispatch jobs for upcoming round.
        for (job_id, worker_ids) in \
            self._next_worker_assignments.items():
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if not is_active:
                continue
            elif (job_id not in self._jobs_with_extended_lease or
                  (job_id in self._jobs_with_extended_lease and
                   job_id in self._completed_jobs_in_current_round)):
                self._logger.info('Dispatching job {0}'.format(job_id))
                self._try_dispatch_job(job_id, worker_ids, next_round=True)

        # Schedule completion events.
        self._schedule_completion_events(round_end_time, pool)

    def _try_dispatch_job(self, job_id, worker_ids, next_round=False):
        """Attempts to dispatch the specified job combination.

           Updates relevant metadata and returns if job has already been
           dispatched.
        """

        # Initialize metadata.
        if not next_round or job_id not in self._current_worker_assignments:
            self._in_progress_updates[job_id] = []
            for single_job_id in job_id.singletons():
                self._lease_update_requests[single_job_id] = []
                self._max_steps[single_job_id] = None

        scale_factor = len(worker_ids)
        worker_type = \
            self._worker_id_to_worker_type_mapping[worker_ids[0]]
        if scale_factor > 1:
            master_addr = self._worker_connections[worker_ids[0]].addr
            master_server_port = self._worker_connections[worker_ids[0]].port
            master_job_ports = []
            for i in range(len(job_id.singletons())):
                master_job_ports.append(BASE_JOB_PORT + self._port_offset)
                self._port_offset += 1
                self._port_offset %= (MAX_PORT - BASE_JOB_PORT)

        # Dispatch the job.
        current_round = self._num_completed_rounds
        if next_round:
            current_round += 1
        for i, worker_id in enumerate(worker_ids):
            job_descriptions = []
            for j, single_job_id in enumerate(job_id.singletons()):
                num_steps = self._jobs[single_job_id].total_steps
                command = self._jobs[single_job_id].command
                # Add distributed args if necessary.
                if scale_factor > 1:
                    command = ('%s --master_addr %s '
                               '--master_port %d '
                               '--world_size %d '
                               '--rank %d' % (command,
                                              master_addr,
                                              master_job_ports[j],
                                              scale_factor, i))
                job_descriptions.append(
                        (single_job_id,
                         command,
                         self._jobs[single_job_id].working_directory,
                         self._jobs[single_job_id].needs_data_dir,
                         self._jobs[single_job_id].num_steps_arg,
                         num_steps))
            self._worker_connections[worker_id].run_job(
                    job_descriptions, worker_id, current_round)
            if not next_round:
                self._remove_available_worker_id(worker_id)

    def _schedule_completion_events(self, round_end_time, pool):
        """Schedules completion events for every dispatched job.

        A completion event in this setting is a callback that will be
        triggered at the conclusion of the current round to indicate that the
        specified job has completed the round. This is necessary for two
        reasons: 1) jobs with extended leases will not trigger the standard
        done_callback at the end of the round, and 2) jobs might freeze.
        """
        current_time = self.get_current_timestamp()
        for job_id in self._current_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if (not is_active or
                job_id in self._completed_jobs_in_current_round):
                continue
            delay = round_end_time - current_time
            if job_id not in self._jobs_with_extended_lease:
                delay += JOB_COMPLETION_BUFFER_TIME
                action = self._kill_job
            else:
                action = self._done_callback_extended_lease
            event = self._completion_event_scheduler.enter(
                    delay=delay, priority=1, action=action, argument=(job_id,))
            self._completion_events[job_id] = event
        pool.submit(self._completion_event_scheduler.run)

    def _end_round(self):
        """Executes final stage of a scheduling round.

        Waits for all currently dispatched jobs to complete, then resets
        the set of jobs with extended leases as well as relevant metadata.
        """

        current_round = self._num_completed_rounds

        # Wait for jobs in current round to complete.
        jobs_to_complete = set()
        for job_id in self._current_worker_assignments:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active:
                jobs_to_complete.add(job_id)
        self._logger.debug('Waiting for following jobs '
                           'to complete: {0}'.format(sorted(jobs_to_complete)))
        while not jobs_to_complete.issubset(
            self._completed_jobs_in_current_round):
            self._scheduler_cv.wait()
            remaining_jobs = jobs_to_complete.difference(
                                self._completed_jobs_in_current_round)
            self._logger.debug('Remaining jobs in round: {0}'.format(
                sorted(remaining_jobs)))
        self._logger.debug(
            'All jobs in round {0} have completed!'.format(current_round))

        if len(self._completion_events) > 0:
            raise RuntimeError('Remaining completion events: {0}'.format(
                self._completion_events.keys()))

        # Reset extended leases.
        jobs_with_extended_lease = list(self._jobs_with_extended_lease)
        for job_id in jobs_with_extended_lease:
            if job_id in self._jobs:
                current_worker_ids = \
                    self._current_worker_assignments[job_id]
                for worker_id in current_worker_ids:
                    self._add_available_worker_id(worker_id)
            self._jobs_with_extended_lease.remove(job_id)
        self._logger.debug('Reset extended leases')

        if not self._is_final_round():
            # The next worker assignments must have been computed here as
            # _end_round is called sequentially after _mid_round.
            if self._next_worker_assignments is None:
                raise RuntimeError(
                    'Next worker assignments have not been computed!')

            for (job_id, worker_ids) in self._next_worker_assignments.items():
                is_active = any([x in self._jobs for x in job_id.singletons()])
                if is_active:
                    # If the job needs to be dispatched again, defer removing
                    # its worker ID.
                    if job_id in self._redispatched_worker_assignments:
                        continue
                    for worker_id in worker_ids:
                        self._remove_available_worker_id(worker_id)

            # Ensure that rounds do not finish earlier than the specified
            # round duration.
            current_time = self.get_current_timestamp()
            round_end_time = \
                self._current_round_start_time + self._time_per_iteration
            remaining_time_in_round = round_end_time - current_time
            if remaining_time_in_round > 0:
                self._logger.debug(
                    'Waiting {0:.2f} seconds before starting '
                    'round {1}...'.format(remaining_time_in_round,
                                          current_round + 1))
                time.sleep(remaining_time_in_round)

        self._num_completed_rounds += 1

        # Reset metadata.
        self._completed_jobs_in_current_round = set()
        self._current_worker_assignments = self._next_worker_assignments
        self._next_worker_assignments = None

        self._scheduler_cv.notifyAll()

        self._logger.info('*** END ROUND {0} ***'.format(current_round))

    def _schedule_with_rounds(self):
        """Schedules jobs on workers using rounds.

        In a loop, schedules in rounds the applications most in need of
        being run (that is, the applications with the highest
        fraction_allocated/fraction_run ratio) using a DP algorithm.
        """

        self._scheduler_cv.acquire()
        # Wait for jobs to arrive and all workers to register with scheduler.
        while (len(self._jobs) == 0 or
                (self._expected_num_workers is not None and
                 len(self._worker_ids) < self._expected_num_workers)):
            self._scheduler_cv.wait()

        # Add all workers to the queue.
        for worker_id in self._worker_ids:
            self._available_worker_ids.put(worker_id)

        # Wait for initial allocation to be computed.
        while self._need_to_update_allocation:
            self._scheduler_cv.wait()

        # Compute initial schedule and dispatch initial set of jobs.
        self._current_worker_assignments = self._schedule_jobs_on_workers()
        state_snapshot = self._get_state_snapshot()
        for (job_id, worker_ids) in self._current_worker_assignments.items():
            self._try_dispatch_job(job_id, worker_ids)
        self._scheduler_cv.release()

        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                is_final_round = self._is_final_round()

                round_start_time = self.get_current_timestamp()
                with self._scheduler_cv:
                    self._begin_round(state_snapshot)

                # Wait for partway through round to recompute schedule.
                delay = self._time_per_iteration * SCHEDULE_RECOMPUTE_FRACTION
                time.sleep(delay)

                with self._scheduler_cv:
                    self._mid_round(pool)
                    state_snapshot = self._get_state_snapshot(deepcopy=True)
                    self._end_round()

                if is_final_round:
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

    def get_num_lease_extensions(self, verbose=True):
        if self._num_lease_extension_opportunities > 0:
            percentage = (100.0 * self._num_lease_extensions) / \
                            self._num_lease_extension_opportunities
            if verbose:
                print('Extended leases {0:.2f}% of the time ({1}/{2})'.format(
                        percentage,
                        self._num_lease_extensions,
                        self._num_lease_extension_opportunities))
        elif verbose:
            percentage = 0
            print('No lease extension opportunities')

        return percentage

    def save_job_timelines(self, timeline_dir):
        if not os.path.isdir(timeline_dir):
            try:
                os.mkdir(timeline_dir)
            except Exception as e:
                self._logger.error('Could not save timelines!')
                traceback.print_exc()
                return

        for job_id in sorted(self._job_timelines.keys()):
            job_dir = os.path.join(timeline_dir, 'job_id={0}'.format(job_id))
            if not os.path.isdir(job_dir):
                os.mkdir(job_dir)
            for i in range(len(self._job_timelines[job_id])):
                timeline_file = os.path.join(job_dir,
                                             'worker={0}.log'.format(i))
                with open(timeline_file, 'w') as f:
                    for event in self._job_timelines[job_id][i]:
                        f.write('{0}\n'.format(event))

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
            self._allocation_changed_since_last_time_reset = True
            self._scheduler_cv.notifyAll()
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
        self._logger.debug('Resetting time run so far')
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
                if job_id not in self._allocation:
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
        self._allocation_changed_since_last_time_reset = False

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
        current_time = self.get_current_timestamp()
        time_since_last_reset = current_time - self._last_reset_time
        reset_interval_elapsed = time_since_last_reset >= \
            self._minimum_time_between_allocation_resets
        need_to_reset_time_run_so_far = \
            reset_interval_elapsed or self._last_reset_time == 0
        if self._simulate:
            need_to_reset_time_run_so_far = \
                (self._need_to_update_allocation and
                 need_to_reset_time_run_so_far)
        else:
            need_to_reset_time_run_so_far = \
                (self._allocation_changed_since_last_time_reset and
                 need_to_reset_time_run_so_far)
        if need_to_reset_time_run_so_far:
            self._reset_time_run_so_far()
            # In simulation mode, wait for allocation computation to complete
            # before proceeding.
            if self._simulate:
                self._allocation = self._compute_allocation()
                self._need_to_update_allocation = False

        # Account for time elapsed since job was dispatched if running on a
        # physical cluster. Note that the total time for each job is the
        # sum of a) the time for all microtasks that have finished
        # (accounted for by self._job_time_so_far), and b) the unaccounted time
        # for all microtasks that are currently running (elapsed_job_time).
        if not self._simulate:
            elapsed_job_time = {}
            elapsed_worker_time = {}
            for job_id in self._current_worker_assignments:
                single_job_id = job_id.singletons()[0]
                if single_job_id not in self._per_job_latest_timestamps:
                    continue
                dispatch_time = self._per_job_latest_timestamps[single_job_id]
                if dispatch_time is None:
                    continue
                dispatch_time = max(dispatch_time, self._last_reset_time)
                elapsed_time = current_time - dispatch_time
                elapsed_job_time[job_id] = {}
                worker_ids = self._current_worker_assignments[job_id]
                worker_type = \
                    self._worker_id_to_worker_type_mapping[worker_ids[0]]
                if worker_type not in elapsed_job_time[job_id]:
                    elapsed_job_time[job_id][worker_type] = 0.0
                if worker_type not in elapsed_worker_time:
                    elapsed_worker_time[worker_type] = 0.0
                elapsed_job_time[job_id][worker_type] += elapsed_time
                elapsed_worker_time[worker_type] += elapsed_time

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        # Compute priorities.
        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            worker_time_so_far = self._worker_time_so_far[worker_type]
            for job_id in self._job_time_so_far:
                worker_time_so_far = self._worker_time_so_far[worker_type]
                if not self._simulate and worker_type in elapsed_worker_time:
                    worker_time_so_far += elapsed_worker_time[worker_type]
                if (worker_time_so_far == 0.0 or
                    worker_type not in self._job_time_so_far[job_id]):
                    fraction = 0.0
                else:
                    job_time_so_far = \
                        self._job_time_so_far[job_id][worker_type]
                    if not self._simulate:
                        if (job_id in elapsed_job_time and
                            worker_type in elapsed_job_time[job_id]):
                            job_time_so_far += \
                                elapsed_job_time[job_id][worker_type]
                    fraction = job_time_so_far / worker_time_so_far
                fractions[worker_type][job_id] = fraction
            for job_id in self._priorities[worker_type]:
                # Don't use inf so 2*new_priority > new_priority.
                #
                # Scale the default value by the allocation so that newly
                # added jobs run according to their respective allocations.
                if job_id not in self._allocation:
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

        if not self._simulate:
            self._logger.debug(
                'Adding worker {0} back to queue...'.format(worker_id))
        self._available_worker_ids.put(worker_id)
        if not self._simulate:
            self._logger.debug(
                'Added worker {0} back to queue'.format(worker_id))

    def _remove_available_worker_id(self, worker_id=None):
        """Returns the worker_id of the next available worker."""

        if self._simulate:
            try:
                return self._available_worker_ids.get_nowait(item=worker_id)
            except queue.Empty as e:
                return None
        else:
            self._logger.debug(
                'Removing worker {0} from the queue...'.format(worker_id))
            ret = self._available_worker_ids.get(item=worker_id)
            if ret != worker_id:
                self._logger.warning(
                    'Worker {0} does not match requested worker {1}!'.format(
                        ret, worker_id))
            self._logger.debug(
                'Removed worker {0} from the queue'.format(ret))
            return ret

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
            self._scheduler_cv.notifyAll()

        return (per_worker_ids, self._time_per_iteration)

    def _init_job_callback(self, job_id):
        """Initializes a job.

           Args:
             job_id: The ID for the (single) job to initialize.
        """
        with self._scheduler_cv:
            # Job could have completed in previous round.
            if job_id not in self._jobs:
                return (0, 0, 0)

            # Wait if this job has been scheduled for the next round
            # but is still running in the previous round (possibly on
            # a different worker).
            while True:
                currently_active = False
                next_job_combination = None

                if self._next_worker_assignments is not None:
                    for job_combination in self._next_worker_assignments:
                        if job_id.overlaps_with(job_combination):
                            next_job_combination = job_combination
                            break

                if next_job_combination is not None:
                    # Check whether this job is blocked by a currently active
                    # job - this could be a job (combination) involving this
                    # job itself or this job's colocation partner in the
                    # upcoming round. For example, consider the following
                    # scenario:
                    #
                    # Round r: Job <0, 1> is scheduled
                    # Round r+1 : Job <1, 2> is scheduled
                    # Job 2 requests initialization partway through round r
                    #
                    # In this case, we would wait to intialize job 2 until
                    # jobs 0 and 1 complete so that job 2 can execute together
                    # with job 1.
                    for job_combination in self._current_worker_assignments:
                        for single_job_id in next_job_combination.singletons():
                            if single_job_id.overlaps_with(job_combination):
                                if (job_combination not in
                                    self._completed_jobs_in_current_round):
                                    currently_active = True
                                    break
                        if currently_active:
                            break

                if currently_active and next_job_combination is not None:
                    self._scheduler_cv.wait()
                else:
                    break

            # Record initializiation as latest job event.
            self._per_job_latest_timestamps[job_id] = \
                self.get_current_timestamp()

            for single_job_id in job_id.singletons():
                self._running_jobs.add(single_job_id)

            # Determine initial lease.
            scale_factor = self._jobs[job_id].scale_factor
            remaining_steps = self._get_remaining_steps(job_id)
            remaining_steps = int(math.ceil(remaining_steps / scale_factor))
            current_time = self.get_current_timestamp()
            current_round_end_time = \
                self._current_round_start_time + self._time_per_iteration
            remaining_time_in_current_round = \
                max(current_round_end_time - current_time, 0)

            # Return a tuple of (steps, duration, extra time) as the initial
            # lease. Extra time is granted if the job was scheduled for the
            # upcoming round but is being initialized in the current round.
            if (self._next_worker_assignments is not None and
                next_job_combination is not None):
                # Job was dispatched early, so add additional time.
                return (remaining_steps, self._time_per_iteration,
                        remaining_time_in_current_round)
            else:
                return (remaining_steps, remaining_time_in_current_round, 0)

    def _update_lease_callback(self, job_id, worker_id, steps, duration,
                               max_steps, max_duration):

        with self._scheduler_lock:
            if job_id not in self._lease_update_requests:
                self._lease_update_requests[job_id] = []
            update_id = len(self._lease_update_requests[job_id])
            self._lease_update_requests[job_id].append((steps, duration,
                                                        max_steps,
                                                        max_duration))

            # Round the remaining steps to the nearest multiple of scale_factor.
            scale_factor = self._jobs[job_id].scale_factor
            remaining_steps = self._get_remaining_steps(job_id)
            remaining_steps = int(math.ceil(remaining_steps / scale_factor))
            current_time = self.get_current_timestamp()
            current_round_end_time = \
                self._current_round_start_time + self._time_per_iteration
            remaining_time_in_current_round = \
                current_round_end_time - current_time
            remaining_time_in_current_round = \
                max(0, remaining_time_in_current_round)

        if steps == 0 or duration == 0:
            return (remaining_steps, remaining_time_in_current_round)

        # Extend the lease if the job has been placed on the same workers
        # for the upcoming round.
        with self._scheduler_lock:
            # TODO: Remove scan of self._jobs_with_extended_lease.
            for job_id_combination in self._jobs_with_extended_lease:
                if job_id.overlaps_with(job_id_combination):
                    updated_lease_duration = duration
                    updated_lease_duration += remaining_time_in_current_round
                    updated_lease_duration += self._time_per_iteration
                    return (max_steps, updated_lease_duration)

        if scale_factor == 1:
            return (max_steps, duration + remaining_time_in_current_round)
        else:
            if update_id == 0:
                assert self._max_steps[job_id] is None

            # The first worker to request a lease update computes the new
            # lease for all workers.
            if update_id == 0:
                with self._scheduler_lock:
                    throughput = steps / duration
                    self._max_steps[job_id] = \
                        min(remaining_steps,
                            steps + int(remaining_time_in_current_round *
                                        throughput))
                    return (self._max_steps[job_id], INFINITY)
            else:
                # Wait for the first update to complete.
                while True:
                    with self._scheduler_lock:
                        max_steps = self._max_steps[job_id]
                        if max_steps is not None:
                            break
                    # TODO: Sleep for less time?
                    self._logger.debug(
                        'Job {0} (worker {1}) waiting for '
                        'lease...'.format(job_id, worker_id))
                    time.sleep(1)
                assert max_steps is not None
                return (max_steps, INFINITY)

    def _kill_job(self, job_id):
        with self._scheduler_cv:
            if job_id not in self._current_worker_assignments:
                raise RuntimeError(
                    'Trying to kill job ({0}) that is not active '
                    'in this round!'.format(job_id))
            elif job_id not in self._completion_events:
                if job_id not in self._completed_jobs_in_current_round:
                    raise RuntimeError(
                        'Completion event for job {0} is not active '
                        'even though job has not completed!'.format(job_id))
                elif job_id not in self._jobs_with_extended_lease:
                    # Job has already completed normally.
                    return
            self._logger.info('Killing job {0}'.format(job_id))
            worker_ids = self._current_worker_assignments[job_id]
            servers = set()
            for worker_id in worker_ids:
                rpc_client = self._worker_connections[worker_id]
                server = (rpc_client.addr, rpc_client.port)
                if server not in servers:
                    for single_job_id in job_id.singletons():
                        self._logger.debug(
                            'Killing job {0} on server {1}:{2}'.format(
                                single_job_id, rpc_client.addr,
                                rpc_client.port))
                        rpc_client.kill_job(single_job_id)
                    servers.add(server)
            del self._completion_events[job_id]

            # Wait for the killed job to send a completion notification and
            # proceed if no notification is sent.
            prev_round = self._num_completed_rounds
            self._logger.debug(
                'Waiting for job {0} to be killed...'.format(job_id))
            self._scheduler_cv.wait(timeout=30)
            self._logger.debug(
                'Checking if job {0} was killed...'.format(job_id))
            new_round = self._num_completed_rounds

            successful_kill = (new_round != prev_round or
                               job_id in self._completed_jobs_in_current_round)
            if successful_kill:
                self._logger.debug(
                    'Job {0} was successfully killed in round {1}'.format(
                        job_id, prev_round))
            else:
                self._logger.debug(
                    'Job {0} was killed but did not complete!'.format(job_id))
                all_worker_ids = set(self._current_worker_assignments[job_id])
                completed_worker_ids = set()
                for update in self._in_progress_updates[job_id]:
                    worker_id = update[0]
                    completed_worker_ids.add(worker_id)
                worker_ids_to_complete = \
                    all_worker_ids.difference(completed_worker_ids)
                self._logger.debug(
                    'Need to send done callbacks for the following '
                    'workers for job {0}: {1}'.format(
                        job_id, worker_ids_to_complete))
        if not successful_kill:
            x = [0 for _ in range(len(job_id.singletons()))]
            for worker_id in worker_ids_to_complete:
                self._logger.debug(
                    'Sending done callback for worker {0} '
                    'for job {1}'.format(worker_id, job_id))
                self._done_callback(job_id, worker_id, x, x)

    def _done_callback_extended_lease(self, job_id):
        kill_job = False

        with self._scheduler_cv:
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if not is_active:
                return

            self._logger.debug('Trying to complete job {0} which had an '
                               'extended lease...'.format(job_id))

            scale_factor = self._jobs[job_id.singletons()[0]].scale_factor
            num_updates = []
            for single_job_id in job_id.singletons():
                num_updates.append(
                    len(self._lease_update_requests[single_job_id]))
            updated_lease = min(num_updates) == scale_factor
            for i, single_job_id in enumerate(job_id.singletons()):
                self._logger.debug('{0} / {1} worker(s) for job {2} have '
                                   'requested a lease update this '
                                   'round'.format(
                                       num_updates[i], scale_factor,
                                       single_job_id))
            if not updated_lease:
                # Job has not requested lease updates so assume it has failed.
                self._logger.error(
                    'Job {0} had an extended lease but has '
                    'been unresponsive'.format(job_id))
                kill_job = True
            elif job_id in self._completion_events:
                self._logger.info('Completing job {0}'.format(job_id))

                # Mark job as completed.
                self._completed_jobs_in_current_round.add(job_id)
                del self._completion_events[job_id]

                # Reset metadata.
                # NOTE: We do not reset self._in_progress_updates here as
                # multi-GPU jobs might have partially completed updates.
                for single_job_id in job_id.singletons():
                    self._lease_update_requests[single_job_id] = []
                    self._max_steps[single_job_id] = None

            if not kill_job:
                self._scheduler_cv.notifyAll()

        if kill_job:
            self._kill_job(job_id)

    def _done_callback(self, job_id, worker_id, all_num_steps,
                       all_execution_times, all_iterator_logs=None):
        """Handles completion of a scheduled job.

        Updates the running total of completed steps and time spent on each
        worker, for every currently active application. Removes the job from
        the scheduler if the job has finished all its requested steps. Adds
        the worker back to the list of available workers.

        Args:
            job_id: The id of the completed job(s).
            worker_id: The id of the worker where the job(s) were completed.
            all_num_steps: List of the number of steps each job ran for.
            all_execution_times: List of the duration each job ran for.
            all_iterator_logs: List of the GavelIterator logs for each job.
        """

        to_remove = []
        with self._scheduler_lock:
            # If current round is r, job might have been dispatched for
            # round r+1 and completed before round r is done. If so,
            # wait for round r to finish before proceeding.
            if not self._simulate:
                while (job_id not in self._current_worker_assignments or
                       job_id in self._completed_jobs_in_current_round):
                    if (job_id not in self._current_worker_assignments and
                        (self._next_worker_assignments is not None and
                         job_id not in self._next_worker_assignments)):
                        self._logger.warning(
                            'Discarding completion notification for job {0} '
                            'as it is not currently scheduled'.format(job_id))
                        return
                    self._logger.debug(
                        'Waiting to complete job {0}...'.format(job_id))
                    self._scheduler_cv.wait()

            # Check whether jobs are still active as jobs might have
            # completed after being dispatched for the subsequent round.
            is_active = {}
            for single_job_id in job_id.singletons():
                is_active[single_job_id] = single_job_id in self._jobs
            if not any(is_active.values()):
                self._logger.info('Job {job_id} (worker {worker_id}) has '
                                  'already completed!'.format(
                                      job_id=job_id, worker_id=worker_id))
                return

            current_timestamp = self.get_current_timestamp()
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            self._add_available_worker_id(worker_id)

            scale_factor = len(self._current_worker_assignments[job_id])
            self._in_progress_updates[job_id].append((worker_id,
                                                      all_num_steps,
                                                      all_execution_times,
                                                      all_iterator_logs))
            if len(self._in_progress_updates[job_id]) < scale_factor:
                return
            else:
                # Sort updates in order of increasing worker ID.
                self._in_progress_updates[job_id].sort(key=lambda x: x[0])

                # If a job completes before the end of the round, cancel the
                # job's completion event.
                self._logger.debug(
                    'Current active completion events: {0}'.format(
                        self._completion_events.keys()))
                if job_id in self._completion_events:
                    event = self._completion_events[job_id]
                    try:
                        self._completion_event_scheduler.cancel(event)
                    except ValueError:
                        # Completion event might have been triggered after
                        # entering done_callback.
                        pass
                    self._logger.debug(
                        'Removing completion event for job {0}'.format(job_id))
                    del self._completion_events[job_id]
                self._completed_jobs_in_current_round.add(job_id)
                micro_task_succeeded = True
                all_worker_ids = \
                    [x[0] for x in self._in_progress_updates[job_id]]
                all_worker_ids.sort()
                all_num_steps = [0] * len(job_id.singletons())
                all_execution_times = [0] * len(job_id.singletons())
                for i, update in enumerate(self._in_progress_updates[job_id]):
                    all_num_steps_ = update[1]
                    all_execution_times_ = update[2]
                    all_iterator_logs_ = update[3]
                    for j, single_job_id in enumerate(job_id.singletons()):
                        if not is_active[single_job_id]:
                            continue
                        elif (all_num_steps_[j] <= 0 or
                              all_execution_times_[j] <= 0):
                            micro_task_succeeded = False
                            break
                    for j, single_job_id in enumerate(job_id.singletons()):
                        all_num_steps[j] += all_num_steps_[j]
                        all_execution_times[j] = max(all_execution_times[j],
                                                     all_execution_times_[j])
                        if all_iterator_logs_ is not None:
                            self._job_timelines[single_job_id][i].extend(
                                all_iterator_logs_[j].split('\n'))

            # Reset metadata.
            self._in_progress_updates[job_id] = []
            for single_job_id in job_id.singletons():
                self._lease_update_requests[single_job_id] = []
                self._max_steps[single_job_id] = None

            if not self._simulate:
                # NOTE: We update the timestamp before calling this
                # function in simulation.
                for single_job_id in job_id.singletons():
                    if is_active[single_job_id]:
                        self._per_job_latest_timestamps[single_job_id] = \
                                self.get_current_timestamp()

            if not micro_task_succeeded:
                # Micro-task failed.
                self._logger.info(
                    '[Micro-task failed]\tJob ID: {job_id}'.format(
                        job_id=job_id))
                if not job_id.is_pair() and is_active[job_id]:
                    self._num_failures_per_job[job_id] += 1
                    if (self._num_failures_per_job[job_id] >=
                        MAX_FAILED_ATTEMPTS):
                        start_time = self._per_job_start_timestamps[job_id]
                        finish_time = self._per_job_latest_timestamps[job_id]
                        duration = finish_time - start_time
                        self._logger.info(
                            '[Job failed]\tJob ID: {job_id}\t'
                            'Start timestamp: {start_timestamp:.2f}\t'
                            'End timestamp: {end_timestamp:.2f}\t'
                            'Duration: {duration:.2f}'.format(
                                job_id=job_id,
                                start_timestamp=start_time,
                                end_timestamp=finish_time,
                                duration=duration))
                        to_remove.append(job_id)
                self._need_to_update_allocation = True

            else:
                self._logger.info(
                    '[Micro-task succeeded]\t'
                    'Job ID: {job_id}\tWorker type: {worker_type}\t'
                    'Worker ID(s): {worker_ids}'.format(
                        job_id=job_id, worker_type=worker_type,
                        worker_ids=str(all_worker_ids)))
                self._num_failures_per_job[job_id] = 0
                for single_job_id, num_steps, execution_time in \
                        zip(job_id.singletons(), all_num_steps,
                            all_execution_times):
                    if not is_active[single_job_id]:
                        self._logger.debug('Job {0} is not active, not '
                                           'updating metadata'.format(
                                               single_job_id))
                        continue
                    if self._per_worker_type_prices is not None:
                        self._job_cost_so_far[single_job_id] += \
                            (self._per_worker_type_prices[worker_type] *
                             execution_time / 3600.0 * scale_factor)
                        job_cost_so_far = \
                            self._job_cost_so_far[single_job_id]
                        self._logger.info(
                            'Job {job_id} cost so far: ${cost:.2f}'.format(
                                job_id=single_job_id, cost=job_cost_so_far))
                    # Job may be multi-GPU, and have already been removed from
                    # running_jobs by another worker.
                    if single_job_id in self._running_jobs:
                        self._running_jobs.remove(single_job_id)
                        self._steps_run_so_far[single_job_id][worker_type] += \
                                num_steps
                        self._total_steps_run[single_job_id] += num_steps
                        remaining_steps = \
                            self._get_remaining_steps(single_job_id)
                        if remaining_steps > 0:
                            if not self._simulate:
                                self._logger.debug(
                                    'Job {job_id} has {steps} '
                                    'remaining steps'.format(
                                        job_id=single_job_id,
                                        steps=remaining_steps))
                        else:
                            start_time = \
                                self._per_job_start_timestamps[single_job_id]
                            finish_time = \
                                self._per_job_latest_timestamps[single_job_id]
                            duration = finish_time - start_time
                            self._logger.info(
                                '[Job succeeded]\tJob ID: {job_id}\t'
                                'Start timestamp: {start_timestamp:.2f}\t'
                                'End timestamp: {end_timestamp:.2f}\t'
                                'Duration: {duration:.2f}'.format(
                                    job_id=single_job_id,
                                    start_timestamp=start_time,
                                    end_timestamp=finish_time,
                                    duration=duration))
                            to_remove.append(single_job_id)

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
                self._remove_job(single_job_id)

            # Schedule the job for re-dispatching if necessary.
            is_active = any([x in self._jobs for x in job_id.singletons()])
            if is_active and job_id in self._jobs_with_extended_lease:
                self._redispatched_worker_assignments[job_id] = \
                    self._next_worker_assignments[job_id]

            self._scheduler_cv.notifyAll()
