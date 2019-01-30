import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc

def run(job_id, command):
  with grpc.insecure_channel('localhost:50052') as channel:
    stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

    request = s2w_pb2.RunRequest(job_id=job_id,
                                 command=command)
    response = stub.Run(request)
