import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rpc_stubs'))

from concurrent import futures
import time
import logging
from multiprocessing.pool import ThreadPool
import subprocess

import grpc

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
import enums_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Job:
  def __init__(self, job_proto):
    self._job_id = job_proto.job_id
    self._command = job_proto.command

  def job_id(self):
    return self._job_id

  def command(self):
    return self._command

class Dispatcher:
  def __init__(self):
    self._thread_pool = ThreadPool()
  
  def launch_job(self, job):
    output = subprocess.check_output(job.command(),
                                     stderr=subprocess.STDOUT,
                                     shell=True)
    print(output)

  def dispatch_job(self, job):
    self._thread_pool.apply_async(self.launch_job, (job,))

class WorkerServer(s2w_pb2_grpc.SchedulerToWorkerServicer):
  def __init__(self):
    self._dispatcher = Dispatcher()

  def _dispatch(self, job_proto):
    self._dispatcher.dispatch_job(Job(job_proto))

  def Run(self, request, context):
    self._dispatch(request)
    return s2w_pb2.RunResponse(job_id=request.job_id,
                               status=enums_pb2.JobStatus.Value('QUEUED'))

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  s2w_pb2_grpc.add_SchedulerToWorkerServicer_to_server(WorkerServer(),
                                                       server)
  server.add_insecure_port('[::]:50052')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__=='__main__':
  logging.basicConfig()
  serve()
