import argparse
import socket
import threading

from runtime.rpc import dispatcher
from runtime.rpc import worker_client
from runtime.rpc import worker_server

WORKER_PORT = 50052

class Worker:
    def __init__(self, sched_ip_addr, sched_port):
        self._worker_ip_addr = socket.gethostbyname(socket.gethostname())
        self._worker_port = WORKER_PORT
        self._worker_rpc_client = worker_client.WorkerRpcClient(
                self._worker_ip_addr, self._worker_port,
                sched_ip_addr, sched_port)
        self._devices = [] # TODO: get devices
        self._worker_id, error = \
            self._worker_rpc_client.register_worker(self._devices)
        if error:
          pass # TODO: handle error
        self._dispatcher = dispatcher.Dispatcher(self._worker_id,
                                                 self._worker_rpc_client)

        callbacks = {
            'Run': self._run_callback,
        }
        self._server_thread = threading.Thread(
                target=worker_server.serve,
                args=(WORKER_PORT, callbacks,))
        self._server_thread.daemon = True
        self._server_thread.start()

    def _run_callback(self, job):
        self._dispatcher.dispatch_job(job)

    def join(self):
        try:
            while True:
                import time
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            return

#TODO: Move this to a separate driver?
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a worker process')
    parser.add_argument('-i', '--ip_addr', type=str, required=True,
                        help='IP address for scheduler server')
    parser.add_argument('-p', '--port', type=int, default=50051,
                        help='Port number for scheduler server')
    args = parser.parse_args()
    opt_dict = vars(args)

    worker = Worker(opt_dict['ip_addr'], opt_dict['port'])
    worker.join()
