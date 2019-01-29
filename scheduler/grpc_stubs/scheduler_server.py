from concurrent import futures
import time
import logging
import threading

import grpc

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import common_pb2
import enums_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Device:
  def __init__(self, worker_id, proto):
    self._worker_id = worker_id
    self._device_id = proto.device_id
    self._type = proto.device_type
    self._available_memory = proto.available_memory

  def name(self):
    device_type = enums_pb2.DeviceType.Name(self._type) 
    return '[Worker: %d | Device: %d | Type: %s]' % (self._worker_id,
                                                     self._device_id,
                                                     device_type)

  def available_memory(self):
    return self._available_memory

class SchedulerServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
  def __init__(self):
    self.lock = threading.Lock()
    self._num_workers = 0
    self._devices = {}

  def _register_new_worker(self):
    with self.lock:
      worker_id = self._num_workers
      self._num_workers += 1
      self._devices[worker_id] = {}
    return worker_id

  def _add_device(self, worker_id, device_proto):
    device_id = device_proto.device_id
    with self.lock:
      if (worker_id in self._devices
          and device_proto.device_id in self._devices[worker_id]):
        device_name = self._devices[worker_id][device_id].name()
        print('Device \'%s\' already registered' % (device_name))
      else:
        self._devices[worker_id][device_id] = Device(worker_id, device_proto)
        device_name = self._devices[worker_id][device_id].name()
        memory = self._devices[worker_id][device_id].available_memory()
        print('Added device %s with %.2f GB available memory' % (device_name,
                                                                 memory))

  def RegisterWorker(self, request, context):
    worker_id = self._register_new_worker()
    for device in request.devices:
      self._add_device(worker_id, device)
    return w2s_pb2.RegisterWorkerResponse(worker_id=worker_id)

  """
  def Heartbeat(self, request, context):
    worker_id = request.worker_id
    device_id = request.device_id
    job_id = request.job_id
    status = enums_pb2.JobStatus.Name(request.status)
    print('[Worker: %d | Device: %d | Job: %d] Status: %s' % (worker_id,
                                                              device_id,
                                                              job_id,
                                                              status))
    return common_pb2.Empty()
  """

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(SchedulerServer(),
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
