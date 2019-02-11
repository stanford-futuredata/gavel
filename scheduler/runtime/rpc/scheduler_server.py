from concurrent import futures
import time

import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import common_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

"""
class SchedulerServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
  def __init__(self, scheduler=None):
    self._scheduler = scheduler

  def Done(self, request, context):
    if self._scheduler is not None:
      self._scheduler._schedule_callback(request.job_id, request.worker_id)
    return common_pb2.Empty()
"""
class SchedulerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self, callbacks):
        print("In server constructor...")
        self._callbacks = callbacks

    def _device_proto_to_device(self, device_proto):
        return None

    def RegisterWorker(self, request, context):
        print('Received RegisterWorker request')
        devices = []
        for device_proto in request.devices:
            devices.append(self._device_proto_to_device(device_proto))
        (worker_id, error) = self._callbacks['RegisterWorker'](devices=devices)
        print('Registered worker %d' % worker_id)
        if error is None:
            print('Returning RegisterWorkerResponse with worker_id %d' % worker_id)
            return w2s_pb2.RegisterWorkerResponse(worker_id=worker_id)
        else:
            return w2s_pb2.RegisterWorkerResponse(error_message=error_message)

    def SendHeartbeat(self, request, context):
        self._callbacks['SendHeartbeat']()
        return common_pb2.Empty()

    def Done(self, request, context):
        self._callbacks['Done'](request.job_id, request.worker_id)
        return common_pb2.Empty()

def serve(port, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
            SchedulerRpcServer(callbacks), server)
    server.add_insecure_port('[::]:%d' % (port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
