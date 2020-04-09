import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import copy
import datetime
import json
import queue
import threading
import time

from job_id_pair import JobIdPair
from job_table import JobTable
from runtime.rpc import scheduler_server, scheduler_client

SERVER_PORT = 50060
INFINITY = 1000000
MULTI_GPU_JOB_TYPES = ['ResNet-18', 'ResNet-50', 'Transformer', 'LM']

class Profiler:
    def __init__(self, num_workers, measurement_time, log_file=None):
        # Profiler parameters.
        self._num_workers = num_workers
        self._measurement_time = measurement_time
        self._log_file = log_file

        # Job metadata.
        self._job_id_counter = 0
        self._job_id_to_job_type = {}
        self._throughputs = {}
        self._scale_factors = {}
        self._num_throughput_updates = {}

        # Worker metadata.
        self._cluster_spec = {}
        self._worker_id_counter = 0
        self._worker_connections = {}
        self._worker_id_to_worker_type = {}
        self._worker_type_to_worker_ids = {}
        self._worker_addrs = {}
        self._worker_queue = queue.Queue()

        # Synchronization lock.
        self._lock = threading.Lock()

        # Logging thread setup.
        self._write_queue = queue.Queue()
        self._logging_thread = threading.Thread(target=self._print_logs)
        self._logging_thread.daemon = True
        self._logging_thread.start()

        # Server thread setup.
        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'Done': self._done_callback,
        }
        self.server_thread = threading.Thread(
            target=scheduler_server.serve,
            args=(SERVER_PORT, callbacks, self._write_queue))
        self.server_thread.daemon = True
        self.server_thread.start()

    """
    =====================================================================
        Utility functions.
    =====================================================================
    """

    def _print_logs(self):
        """Print logging information."""
        while True:
            output = self._write_queue.get()
            if self._log_file is not None:
                with open(self._log_file, 'a') as f:
                    f.write('[%s] %s' % (str(datetime.datetime.now()),
                                         output))
            else:
                print('[%s] %s' % (str(datetime.datetime.now()),
                                   output))

    def _initialize_throughputs(self, worker_type=None, job_type=None):
        """Initialize throughputs data structure."""
        if worker_type is not None:
            self._throughputs[worker_type] = {}
            for job_description in JobTable:
                job_type = job_description.model
                self._throughputs[worker_type][job_type] = {}
        elif job_type is not None:
            for worker_type in self._throughputs:
                self._throughputs[worker_type][job_type] = {}

    def _initialize_task(self, job_descriptions, scale_factor=1):
        """Initialize a task to submit to the worker."""
        task = []
        for job_description in job_descriptions:
            with self._lock:
                job_id = JobIdPair(self._job_id_counter, None)
                self._job_id_counter += 1
                job_type = job_description.model
                if scale_factor > 1:
                    job_type += ' (scale factor %d)' % (scale_factor)
                    self._initialize_throughputs(job_type=job_type)
                self._job_id_to_job_type[job_id] = job_type
            task.append([job_id, job_description.command,
                         job_description.needs_data_dir,
                         job_description.num_steps_arg,
                         INFINITY])
        if len(job_descriptions) > 1:
            merged_job_id = JobIdPair(task[0][0][0], task[1][0][0])
        else:
            merged_job_id = task[0][0]
        self._scale_factors[merged_job_id] = scale_factor
        return task

    def _can_be_run_multi_gpu(self, job_type):
        """Returns True if the job type supports distributed execution."""
        for multi_gpu_job_type in MULTI_GPU_JOB_TYPES:
            if multi_gpu_job_type in job_type:
                return True
        return False

    def _initialize_per_worker_type_task_queues(self, isolated,
                                                packing, distributed):
        """Initializes the task queues for every worker type."""
        per_worker_type_task_queues = {}
        for worker_type in self._cluster_spec:
            per_worker_type_task_queues[worker_type] = queue.Queue()

        scale_factor = 1
        while scale_factor <= max(self._cluster_spec.values()):
            if isolated:
                for worker_type in per_worker_type_task_queues:
                    if scale_factor > self._cluster_spec[worker_type]:
                        continue
                    for i in range(len(JobTable)):
                        job_type = JobTable[i].model
                        if (scale_factor > 1 and
                            not self._can_be_run_multi_gpu(job_type)):
                            continue
                        task = self._initialize_task(
                                [JobTable[i]], scale_factor=scale_factor)
                        per_worker_type_task_queues[worker_type].put(
                                    (task, scale_factor))

            if packing:
                for worker_type in per_worker_type_task_queues:
                    if scale_factor > self._cluster_spec[worker_type]:
                        continue
                    for i in range(len(JobTable)):
                        if (scale_factor > 1 and
                            not self._can_be_run_multi_gpu(JobTable[i].model)):
                            continue
                        for j in range(i, len(JobTable)):
                            if (scale_factor > 1 and         
                                not self._can_be_run_multi_gpu(JobTable[j].model)):
                                continue
                            task = self._initialize_task(
                                    [JobTable[i], JobTable[j]],
                                    scale_factor=scale_factor)
                            per_worker_type_task_queues[worker_type].put(
                                    (task, scale_factor))
            if distributed:
                scale_factor *= 2
            else:
                break

        return per_worker_type_task_queues

    def _wait_for_workers(self):
        """Wait for the expected number of workers to register."""
        while True:
            with self._lock:
                num_workers = len(self._worker_connections.keys())
            if num_workers < self._num_workers:
                time.sleep(5)
            else:
                break

    def _get_job_types_from_job_id(self, job_id):
        """Returns a list of the job types associated with the job ID."""
        job_types = []
        if job_id.is_pair():
            for single_job_id in job_id.singletons():
                job_types.append(self._job_id_to_job_type[single_job_id])
        else:
            job_types.append(self._job_id_to_job_type[job_id])
            job_types.append(None)
        return job_types

    def _shutdown_workers(self):
        """Sends a shutdown message to all workers."""
        for worker_id in self._worker_connections:
            self._worker_connections[worker_id].shutdown()

    """
    =====================================================================
        RPC callbacks.
    =====================================================================
    """

    def _register_worker_callback(self, worker_type, num_gpus, ip_addr, port):
        """Registers a new worker."""
        rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)
        per_worker_ids = []
        with self._lock:
            if worker_type not in self._cluster_spec:
                self._cluster_spec[worker_type] = 0
                self._throughputs[worker_type] = {}
                self._worker_type_to_worker_ids[worker_type] = []
                self._initialize_throughputs(worker_type=worker_type)
            self._cluster_spec[worker_type] += num_gpus
            for i in range(num_gpus):
                worker_id = self._worker_id_counter
                self._worker_id_counter += 1
                per_worker_ids.append(worker_id)
                self._worker_id_to_worker_type[worker_id] = worker_type
                self._worker_type_to_worker_ids[worker_type].append(worker_id)
                self._worker_addrs[worker_id] = (ip_addr, port)
                self._worker_connections[worker_id] = rpc_client
                self._write_queue.put(
                    'Registered worker %d (%s) at %s:%s' % (worker_id,
                                                            worker_type,
                                                            ip_addr,
                                                            port))
        return (per_worker_ids, self._measurement_time)

    def _done_callback(self, job_id, worker_id, all_num_steps,
                       all_execution_times):
        """Updates the throughput of the associated job(s)."""
        with self._lock:
            worker_type = self._worker_id_to_worker_type[worker_id]
            job_types = self._get_job_types_from_job_id(job_id)
            all_throughputs = self._throughputs[worker_type]
            scale_factor = self._scale_factors[job_id]
            job_throughputs = []
            for (num_steps, execution_time) in \
                zip(all_num_steps, all_execution_times):
                if execution_time <= 0:
                    job_throughputs.append(0)
                else:
                    job_throughputs.append(num_steps / execution_time)

            # Initialize/reset throughputs if necessary.
            job_failed = min(all_execution_times) <= 0
            if job_failed or job_types[1] not in all_throughputs[job_types[0]]:
                if job_id.is_pair():
                    all_throughputs[job_types[0]][job_types[1]] = [0.0, 0.0]
                    all_throughputs[job_types[1]][job_types[0]] = [0.0, 0.0]
                else:
                    all_throughputs[job_types[0]][job_types[1]] = 0.0

            # Update throughputs.
            if job_failed:
                self._num_throughput_updates[job_id] = scale_factor
            else:
                if job_id.is_pair():
                    for i in range(len(job_throughputs)):
                        all_throughputs[job_types[0]][job_types[1]][i] += \
                            job_throughputs[i]
                        all_throughputs[job_types[1]][job_types[0]][1-i] += \
                            job_throughputs[i]
                else:
                    all_throughputs[job_types[0]][job_types[1]]+= \
                        job_throughputs[0]
                self._num_throughput_updates[job_id] += 1

            # Print logging information.
            if self._num_throughput_updates[job_id] == scale_factor:
                updated_throughputs = \
                    all_throughputs[job_types[0]][job_types[1]]
                throughputs_str = str(updated_throughputs)
                if job_id.is_pair():
                    self._write_queue.put(
                        'Throughputs for %s on %s: %s' % (str(job_types),
                                                          worker_type,
                                                          throughputs_str))
                else:
                    assert(job_types[1] is None)
                    self._write_queue.put(
                        'Throughput for %s on %s: %s' % (job_types[0],
                                                         worker_type,
                                                         throughputs_str))
        self._worker_queue.get()

    """
    =====================================================================
        Public API functions.
    =====================================================================
    """

    def profile(self, isolated, packed, distributed):
        """Profiles the job types in the desired configuration(s)."""
        self._wait_for_workers()

        per_worker_type_task_queues = \
            self._initialize_per_worker_type_task_queues(isolated,
                                                         packed,
                                                         distributed)

        done = False
        while not done:
            num_tasks = 0
            for worker_type in per_worker_type_task_queues:
                num_tasks += per_worker_type_task_queues[worker_type].qsize()
            assert(num_tasks > 0)
            for worker_type in self._cluster_spec:
                worker_ids = self._worker_type_to_worker_ids[worker_type]
                worker_id_ptr = 0
                num_remaining_worker_ids = len(worker_ids)
                unschedulable_queue = queue.Queue()
                while (num_remaining_worker_ids > 0 and
                        not per_worker_type_task_queues[worker_type].empty()):
                    (task, scale_factor) = \
                        per_worker_type_task_queues[worker_type].get()
                    if scale_factor > num_remaining_worker_ids:
                        unschedulable_queue.put((task, scale_factor))
                        continue
                    if len(task) > 1:
                        merged_job_id = JobIdPair(task[0][0][0], task[1][0][0])
                    else:
                        merged_job_id = task[0][0]
                    self._num_throughput_updates[merged_job_id] = 0
                    for i in range(worker_id_ptr, worker_id_ptr+scale_factor):
                        worker_id = worker_ids[i]
                        for j, job_description in enumerate(task):
                            job_id = job_description[0]
                            job_type = self._job_id_to_job_type[job_id]
                            self._write_queue.put(
                                    'Scheduling job %s (%s) on '
                                    'worker %d (%s)' % (job_id,
                                                        job_type,
                                                        worker_id,
                                                        worker_type))
                            if scale_factor > 1:
                                if i == worker_id_ptr:
                                    if j == 0:
                                        base_commands = []
                                    base_commands.append(job_description[1])
                                command = copy.deepcopy(base_commands[j])
                                master_id = worker_ids[worker_id_ptr]
                                (master_addr, master_port) = \
                                    self._worker_addrs[master_id]
                                world_size = scale_factor
                                rank = i - worker_id_ptr
                                command = ('%s --master_addr %s '
                                           '--master_port %d '
                                           '--world_size %d '
                                           '--rank %d') % (command,
                                                           master_addr,
                                                           master_port+1+master_id+j,
                                                           world_size,
                                                           rank)
                                task[j][1] = command
                        self._worker_queue.put(worker_id)
                        self._worker_connections[worker_id].run(task,
                                                                worker_id)
                    worker_id_ptr += scale_factor
                    num_remaining_worker_ids -= scale_factor
                while not unschedulable_queue.empty():
                    (task, scale_factor) = unschedulable_queue.get()
                    per_worker_type_task_queues[worker_type].put(
                            (task, scale_factor))
            while not self._worker_queue.empty():
                time.sleep(2)
            done = True
            for worker_type in per_worker_type_task_queues:
                if not per_worker_type_task_queues[worker_type].empty():
                    done = False
                    break

        self._shutdown_workers()

    def output(self, output_file):
        """Outputs the throughputs to a file in JSON format."""
        with open(output_file, 'w') as f:
            f.write(json.dumps(self._throughputs, indent=4))

def main(args):
    if not args.isolated and not args.packed and not args.distributed:
        raise ValueError('At least one of "--isolated", "--packed", and '
                         '"--distributed" must be set')

    profiler = Profiler(args.num_workers, args.measurement_time)
    profiler.profile(args.isolated, args.packed, args.distributed)
    profiler.output(args.output_file)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Measure throughputs')
    parser.add_argument('-n', '--num_workers', type=int, required=True,
                        help='Number of workers')
    parser.add_argument('-m', '--measurement_time', type=int, default=150,
                        help='Time per measurement in seconds')
    parser.add_argument('-l', '--log_file', type=str, default=None,
                        help='Log file')
    parser.add_argument('-i', '--isolated', action='store_true',
                        help='Measure isolated throughputs')
    parser.add_argument('-p', '--packed', action='store_true',
                        help='Measure packed throughputs')
    parser.add_argument('-d', '--distributed', action='store_true',
                        help='Measure distributed throughputs')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='JSON output file for throughputs')
    args = parser.parse_args()
    main(args)
