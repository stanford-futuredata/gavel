from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rpc_stubs'))

import argparse
import logging

import grpc

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import enums_pb2

def run(job_id, command):
  with grpc.insecure_channel('localhost:50052') as channel:
    stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

    request = s2w_pb2.RunRequest(job_id=job_id,
                                 command=command)
    response = stub.Run(request)
