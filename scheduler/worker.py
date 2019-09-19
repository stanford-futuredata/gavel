import argparse
import socket
import sys
import threading

import job_id_pair

from runtime.rpc import dispatcher
from runtime.rpc import worker_client
from runtime.rpc import worker_server

class Worker:
    def __init__(self, worker_type, sched_ip_addr, sched_port, worker_port,
                 gpu_id, time_per_iteration):
        self._gpu_id = gpu_id
        self._worker_type = worker_type
        self._worker_ip_addr = socket.gethostbyname(socket.gethostname())
        self._worker_port = worker_port
        self._worker_rpc_client = worker_client.WorkerRpcClient(
                self._worker_type, self._worker_ip_addr,
                self._worker_port, sched_ip_addr, sched_port)
        self._devices = [] # TODO: get devices
        self._results = {}
        self._single_job_id_to_job_id_pair_map = {}
        self._time_per_iteration = time_per_iteration

        print('Starting server at port %d' % (worker_port))

        callbacks = {
            'Run': self._run_callback,
            'Shutdown': self._shutdown_callback,
        }

        self._server_thread = threading.Thread(
            target=worker_server.serve,
            args=(worker_port, callbacks,))
        self._server_thread.daemon = True
        self._server_thread.start()

        self._worker_id, error = \
            self._worker_rpc_client.register_worker(self._devices)
        if error:
            raise RuntimeError(error)

        self._dispatcher = dispatcher.Dispatcher(self._worker_id,
                                                 self._gpu_id,
                                                 self._done_callback,
                                                 self._time_per_iteration)

        self._server_thread.join()

    def _run_callback(self, jobs):
        # hack to prevent a job being dispatched before the dispatcher is set up
        # TODO: fix this by sending a "I'm ready" message to scheduler
        while True:
            try:
                self._dispatcher
                break
            except Exception as e:
              continue
        if len(jobs) == 1:
            job_id = job_id_pair.JobIdPair(jobs[0].job_id, None)
        elif len(jobs) == 2:
            job_id = job_id_pair.JobIdPair(jobs[0].job_id, jobs[1].job_id)
        self._results[job_id] = []
        for job in jobs:
            self._single_job_id_to_job_id_pair_map[job.job_id] = \
                job_id
            self._dispatcher.dispatch_job(job)

    def _get_steps(self, outputs):
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

        all_steps = None
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
            steps = int(steps) - start_steps
            if all_steps is None:
                all_steps = (steps,)
            else:
                all_steps += (steps,)
        return all_steps

    def _done_callback(self, single_job_id, worker_id, num_steps,
                       execution_time, output):
        job_id = self._single_job_id_to_job_id_pair_map[single_job_id]
        self._results[job_id].append([single_job_id, worker_id,
                                      num_steps, execution_time, output])
        if ((len(self._results[job_id]) == 1 and job_id[1] is None) or
            (len(self._results[job_id]) == 2)):
            self._results[job_id].sort(key=lambda x: x[0])
            outputs = []
            for i in range(len(self._results[job_id])):
              outputs.append(self._results[job_id][i][-1])
            steps = self._get_steps(outputs)
            for i in range(len(steps)):
              self._results[job_id][i][2] = steps[i]
            self._worker_rpc_client.notify_scheduler(self._results[job_id])
            del self._results[job_id]
            del self._single_job_id_to_job_id_pair_map[single_job_id]

    def _shutdown_callback(self):
        self._dispatcher.shutdown()

    def join(self):
        self._server_thread.join()

#TODO: Move this to a separate driver?
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a worker process')
    parser.add_argument('-t', '--worker_type', type=str, required=True,
                        help='Worker type')
    parser.add_argument('-i', '--ip_addr', type=str, required=True,
                        help='IP address for scheduler server')
    parser.add_argument('-s', '--sched_port', type=int, default=50060,
                        help='Port number for scheduler server')
    parser.add_argument('-w', '--worker_port', type=int, default=50061,
                        help='Port number for worker server')
    parser.add_argument('-g', '--gpu_id', type=int, required=True,
                        help='GPU ID')
    parser.add_argument('--time_per_iteration', type=int, required=True,
                        help='Time given to each scheduler round')
    args = parser.parse_args()
    opt_dict = vars(args)

    worker = Worker(opt_dict['worker_type'], opt_dict['ip_addr'],
                    opt_dict['sched_port'], opt_dict['worker_port'],
                    opt_dict['gpu_id'], opt_dict['time_per_iteration'])
