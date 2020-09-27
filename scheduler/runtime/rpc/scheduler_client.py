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
        self._addr = server_ip_addr
        self._port = port
        self._server_loc = '%s:%d' % (server_ip_addr, port)

    @property
    def addr(self):
        return self._addr

    @property
    def port(self):
        return self._port

    def run(self, job_descriptions, worker_id, round_id):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            request = s2w_pb2.RunRequest()
            for (job_id, command, working_directory, needs_data_dir,
                 num_steps_arg, num_steps) in job_descriptions:
                job_description = request.job_descriptions.add()
                job_description.job_id = job_id[0] # job_id is a JobIdPair
                job_description.command = command
                job_description.working_directory = working_directory
                job_description.needs_data_dir = needs_data_dir
                job_description.num_steps_arg = num_steps_arg
                job_description.num_steps = num_steps
            request.worker_id = worker_id
            request.round_id = round_id
            response = stub.Run(request)

    def reset(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            response = stub.Reset(common_pb2.Empty())

    def shutdown(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            response = stub.Shutdown(common_pb2.Empty())
