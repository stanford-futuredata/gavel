import argparse
import socket
import sys
import threading

from runtime.rpc import dispatcher
from runtime.rpc import worker_client
from runtime.rpc import worker_server

class Worker:
    def __init__(self, worker_type, sched_ip_addr, sched_port, worker_port,
                 gpu_id):
        self._gpu_id = gpu_id
        self._worker_type = worker_type
        self._worker_ip_addr = socket.gethostbyname(socket.gethostname())
        self._worker_port = worker_port
        self._worker_rpc_client = worker_client.WorkerRpcClient(
                self._worker_type, self._worker_ip_addr,
                self._worker_port, sched_ip_addr, sched_port)
        self._devices = [] # TODO: get devices

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

        self._worker_id, self._round_duration, error = \
            self._worker_rpc_client.register_worker(self._devices)
        if error:
            raise RuntimeError(error)
        
        self._dispatcher = dispatcher.Dispatcher(self._worker_id,
                                                 self._round_duration,
                                                 self._gpu_id,
                                                 self._worker_rpc_client)
        
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
        self._dispatcher.dispatch_jobs(jobs)

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
    args = parser.parse_args()
    opt_dict = vars(args)

    worker = Worker(opt_dict['worker_type'], opt_dict['ip_addr'],
                    opt_dict['sched_port'], opt_dict['worker_port'],
                    opt_dict['gpu_id'])
