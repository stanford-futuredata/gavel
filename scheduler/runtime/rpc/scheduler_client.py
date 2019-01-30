import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc

def register_worker(worker_id):
  with grpc.insecure_channel('localhost:50052') as channel:
    stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

    request = s2w_pb2.RegisterWorkerRequest(worker_id=worker_id)
    response = stub.RegisterWorker(request)

def run(job_id, command, num_epochs):
  with grpc.insecure_channel('localhost:50052') as channel:
    stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

    request = s2w_pb2.RunRequest(job_id=job_id,
                                 command=command,
                                 num_epochs=num_epochs)
    response = stub.Run(request)
