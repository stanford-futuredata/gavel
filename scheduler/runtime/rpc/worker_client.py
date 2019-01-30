from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rpc_stubs'))

import logging

import grpc

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import enums_pb2

def notify_scheduler(job_id, worker_id):
  with grpc.insecure_channel('localhost:50051') as channel:
    stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
    
    # Send a Done message.
    request = w2s_pb2.DoneRequest(job_id=job_id, worker_id=worker_id)
    response = stub.Done(request)
    print('Notified scheduler of completion of %s on %s' % (job_id, worker_id))
