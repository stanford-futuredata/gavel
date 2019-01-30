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

def run(command):
  with grpc.insecure_channel('localhost:50052') as channel:
    stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)

    request = s2w_pb2.RunRequest(job_id=0,
                                 command=command)
    response = stub.Run(request)
    print('Job %d has status %s' % (response.job_id,
                                    enums_pb2.JobStatus.Name(response.status)))

if __name__=='__main__':
  logging.basicConfig()
  
  parser = argparse.ArgumentParser(
      description='Remotely schedule a job on a worker')
  parser.add_argument('-c', '--command', type=str,
                      default='echo "Hello world!"')

  args = parser.parse_args()
  opt_dict = vars(args)
  command = opt_dict['command']

  run(command)
