from concurrent import futures
import time
from multiprocessing.pool import ThreadPool
import subprocess

import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
import enums_pb2


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Job:
  def __init__(self, job_proto):
    self._job_id = job_proto.job_id
    self._command = job_proto.command
    self._num_epochs = job_proto.num_epochs

  def job_id(self):
    return self._job_id

  def command(self):
    return self._command

  def num_epochs(self):
    return self._num_epochs

class Dispatcher:
  def __init__(self, worker_id):
    self._thread_pool = ThreadPool()
    self._worker_id = worker_id
  
  def launch_job(self, job):
    import worker_client
    output = subprocess.check_output(job.command(),
                                     stderr=subprocess.STDOUT,
                                     shell=True)
    print("Job ID: %d, Command: '%s', Num_epochs: %d, Output:" % (
          job.job_id(), job.command(), job.num_epochs()), output)
    worker_client.notify_scheduler(job.job_id(), self._worker_id)

  def dispatch_job(self, job):
    self._thread_pool.apply_async(self.launch_job, (job,))

class WorkerServer(s2w_pb2_grpc.SchedulerToWorkerServicer):
  def __init__(self):
    self._dispatcher = None

  def _dispatch(self, job_proto):
    assert self._dispatcher is not None
    self._dispatcher.dispatch_job(Job(job_proto))

  def RegisterWorker(self, request, context):
    self._worker_id = request.worker_id
    self._dispatcher = Dispatcher(self._worker_id)
    return common_pb2.Empty()

  def Run(self, request, context):
    self._dispatch(request)
    return common_pb2.Empty()

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


if __name__ == '__main__':
  serve()
