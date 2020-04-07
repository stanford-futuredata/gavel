import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
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
THROUGHPUT_ESTIMATION_INTERVAL = 10

class Profiler:
    def __init__(self, num_workers, measurement_time, log_file=None):
        self._num_workers = num_workers
        self._measurement_time = measurement_time
        self._log_file = log_file
        # TODO: Make this configurable?
        self._throughput_estimation_interval = THROUGHPUT_ESTIMATION_INTERVAL
        self._job_id_counter = 0
        self._job_id_to_job_type = {}
        self._cluster_spec = {}
        self._worker_id_counter = 0
        self._worker_connections = {}
        self._worker_id_to_worker_type = {}
        self._worker_type_to_worker_ids = {}
        self._write_queue = queue.Queue()
        self._throughputs = {}
        self._lock = threading.Lock()

        self._logging_thread = threading.Thread(target=self._print_logs)
        self._logging_thread.daemon = True
        self._logging_thread.start()

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
        while True:
            output = self._write_queue.get()
            if self._log_file is not None:
                with open(self._log_file, 'a') as f:
                    f.write('[%s] %s' % (str(datetime.datetime.now()),
                                         output))
            else:
                print('[%s] %s' % (str(datetime.datetime.now()),
                                   output))

    def _initialize_throughputs(self, worker_type):
        self._throughputs[worker_type] = {}
        for job_description in JobTable:
            job_type = job_description.model
            self._throughputs[worker_type][job_type] = {}

    def _initialize_task(self, job_descriptions):
        # TODO: Support distributed.
        task = []
        for job_description in job_descriptions:
            with self._lock:
                job_id = JobIdPair(self._job_id_counter, None)
                self._job_id_counter += 1
                self._job_id_to_job_type[job_id] = job_description.model
            interval = self._throughput_estimation_interval
            command = '%s --throughput_estimation_interval %d' % (
                    job_description.command, THROUGHPUT_ESTIMATION_INTERVAL)
            task.append((job_id, command,
                         job_description.needs_data_dir,
                         job_description.num_steps_arg,
                         INFINITY))
        return task

    def _initialize_per_worker_type_task_queues(self, isolated,
                                                packing, distributed):
        per_worker_type_task_queues = {}
        for worker_type in self._cluster_spec:
            per_worker_type_task_queues[worker_type] = queue.Queue()

        for worker_type in per_worker_type_task_queues:
            if isolated:
                for i in range(len(JobTable)):
                    task = self._initialize_task([JobTable[i]])
                    per_worker_type_task_queues[worker_type].put(task)

            if packing:
                for i in range(len(JobTable)):
                    for j in range(i, len(JobTable)):
                        task = self._initialize_task([JobTable[i],
                                                      JobTable[j]])
                        per_worker_type_task_queues[worker_type].put(task)

        # TODO: Support distributed.
        if distributed:
            pass

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
        job_types = []
        if job_id.is_pair():
            for single_job_id in job_id.singletons():
                job_types.append(self._job_id_to_job_type[single_job_id])
        else:
            job_types.append(self._job_id_to_job_type[job_id])
            job_types.append(None)
        return job_types
    
    def _get_throughputs(self, outputs):
        earliest_end_time = None
        for output in outputs:
            lines = output.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if '[THROUGHPUT_ESTIMATION]' in lines[i]:
                    _, time, _ = lines[i].split('\t')
                    if (earliest_end_time is None or
                        float(time) < earliest_end_time):
                        earliest_end_time = float(time)
                    break

        throughputs = None
        for output in outputs:
            lines = output.split('\n')
            start_time = None
            for line in lines:
                if '[THROUGHPUT_ESTIMATION]' in line:
                    _, time, steps = line.split('\t')
                    if start_time is None:
                        start_time = float(time)
                        start_steps = int(steps)
                    elif float(time) > earliest_end_time:
                        break
            if start_time is None:
                return (-1, -1)
            throughput =\
                (int(steps) - start_steps) / (float(time) - start_time)
            if throughputs is None:
                throughputs = (throughput,)
            else:
                throughputs += (throughput,)

        return throughputs


    """
    =====================================================================
        RPC callbacks.
    =====================================================================
    """

    def _register_worker_callback(self, worker_type, num_gpus, ip_addr, port):
        rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)
        per_worker_ids = []
        with self._lock:
            if worker_type not in self._cluster_spec:
                self._cluster_spec[worker_type] = 0
                self._throughputs[worker_type] = {}
                self._worker_type_to_worker_ids[worker_type] = []
                self._initialize_throughputs(worker_type)
            self._cluster_spec[worker_type] += num_gpus
            for i in range(num_gpus):
                worker_id = self._worker_id_counter
                self._worker_id_counter += 1
                per_worker_ids.append(worker_id)
                self._worker_id_to_worker_type[worker_id] = worker_type
                self._worker_type_to_worker_ids[worker_type].append(worker_id)
                self._worker_connections[worker_id] = rpc_client
                self._write_queue.put(
                    'Registered worker %d (%s) at %s:%s' % (worker_id,
                                                            worker_type,
                                                            ip_addr,
                                                            port))
        return (per_worker_ids, self._measurement_time)

    def _done_callback(self, job_id, worker_id, all_num_steps,
                       all_execution_times, all_outputs):
        with self._lock:
            worker_type = self._worker_id_to_worker_type[worker_id]
            job_types = self._get_job_types_from_job_id(job_id)
            throughputs = self._get_throughputs(all_outputs)
            if job_id.is_pair():
                self._throughputs[worker_type][job_types[0]][job_types[1]] =\
                    throughputs
                self._throughputs[worker_type][job_types[1]][job_types[0]] =\
                    throughputs
                self._write_queue.put(
                    'Throughputs for %s on %s: %s' % (str(job_types),
                                                      worker_type,
                                                      str(throughputs)))
            else:
                assert(job_types[1] is None)
                self._throughputs[worker_type][job_types[0]][job_types[1]] =\
                    throughputs[0]
                self._write_queue.put(
                    'Throughput for %s on %s: %f' % (job_types[0], worker_type,
                                                     throughputs[0]))
        self._worker_queue.put(worker_id)

    """
    =====================================================================
        Public API functions.
    =====================================================================
    """

    def profile(self, isolated, packed, distributed):
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
            with self._lock:
                self._worker_queue = queue.Queue(min(self._num_workers,
                                                     num_tasks))
            for worker_type in self._cluster_spec:
                worker_ids = self._worker_type_to_worker_ids[worker_type]
                for worker_id in worker_ids:
                    task = per_worker_type_task_queues[worker_type].get()
                    for job_description in task:
                        job_id = job_description[0]
                        job_type = self._job_id_to_job_type[job_id]
                        self._write_queue.put('Scheduling job %s (%s) on '
                                              'worker %d (%s)' % (job_id,
                                                                  job_type,
                                                                  worker_id,
                                                                  worker_type))
                    self._worker_connections[worker_id].run(
                            task, worker_id, request_output=True)
            while not self._worker_queue.full():
                time.sleep(2)
            done = True
            for worker_type in per_worker_type_task_queues:
                if not per_worker_type_task_queues[worker_type].empty():
                    done = False
                    break

    def output(self, output_file):
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
