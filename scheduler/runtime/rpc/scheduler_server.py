import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rpc_stubs'))

from concurrent import futures
import time
import logging

import grpc

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import common_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SchedulerServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
  def __init__(self, scheduler=None):
    self._scheduler = scheduler

  def Done(self, request, context):
    if self._scheduler is not None:
      self._scheduler._schedule_callback(request.job_id, request.worker_id)
    return common_pb2.Empty()


def serve(scheduler):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(SchedulerServer(scheduler),
                                                       server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)
