from __future__ import print_function
import logging

import grpc

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import enums_pb2

def run():
  with grpc.insecure_channel('localhost:50051') as channel:
    stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
    
    # Send a RegisterWorker message
    devices = []
    # TODO(keshav2): Send actual GPU devices
    devices.append(
      w2s_pb2.RegisterWorkerRequest.Device(
        device_id=0,
        device_type=enums_pb2.DeviceType.Value('V100'),
        available_memory=16.0))
    request = w2s_pb2.RegisterWorkerRequest(devices=devices)
    response = stub.RegisterWorker(request)
    worker_id = response.worker_id
    print('Registered worker %d' % (worker_id))

    """
    # Send a Heartbeat message 
    request = \
      w2s_pb2.HeartbeatRequest(worker_id=worker_id,
                               device_id=0,
                               job_id=0,
                               status=enums_pb2.JobStatus.Value('RUNNING'))
    stub.Heartbeat(request)
    """

if __name__=='__main__':
  logging.basicConfig()
  run()
