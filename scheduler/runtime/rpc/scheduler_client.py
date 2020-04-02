import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2 

class SchedulerRpcClient:
    """Scheduler client for sending RPC requests to a worker server."""

    def __init__(self, server_ip_addr, port):
        self._server_loc = '%s:%d' % (server_ip_addr, port)

    def run(self, job_descriptions):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

            request = s2w_pb2.RunRequest()
            for (job_id, command, needs_data_dir,
                 num_steps_arg, num_steps) in job_descriptions:
                job_description = request.job_descriptions.add()
                job_description.job_id = job_id
                job_description.command = command
                job_description.needs_data_dir = needs_data_dir
                job_description.num_steps_arg = num_steps_arg
                job_description.num_steps = num_steps
            response = stub.Run(request)


    def shutdown(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            stub.Shutdown(common_pb2.Empty())
