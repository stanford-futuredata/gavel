import threading

from runtime.rpc import dispatcher
from runtime.rpc import worker_client
from runtime.rpc import worker_server

class Worker:
    def __init__(self, port):
        self._worker_rpc_client = worker_client.WorkerRpcClient("localhost",
                                                                50051)
        self._dispatcher = dispatcher.Dispatcher(self._worker_rpc_client)
        callbacks = {
                'Run': self._dispatch,
            }
        self._server_thread = threading.Thread(
                target=worker_server.serve,
                args=(port, callbacks,))
        self._server_thread.daemon = True
        self._server_thread.start()

    def _dispatch(self, job):
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
    worker = Worker(50052)
    worker.join()
