import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc

class SchedulerRpcClient:
    """Scheduler client for sending RPC requests to a worker server."""

    def __init__(self, server_ip_addr, port):
        self._server_loc = '%s:%d' % (server_ip_addr, port)

    def run(self, job_id, command, num_epochs):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

            request = s2w_pb2.RunRequest(job_id=job_id,
                                         command=command,
                                         num_epochs=num_epochs)
            response = stub.Run(request)
