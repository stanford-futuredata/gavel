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
  def __init__(self, queue=None):
    self._queue = queue

  def Done(self, request, context):
    if self._queue is not None:
        self._queue.add(request.worker_id)
    return common_pb2.Empty()


def serve(queue):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(SchedulerServer(queue),
                                                       server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__=='__main__':
  logging.basicConfig()
  serve()
