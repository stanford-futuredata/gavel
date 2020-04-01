from concurrent import futures
import time

import grpc
import os
import sys
import socket
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import common_pb2
from job_id_pair import JobIdPair

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class SchedulerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self, callbacks):
        self._callbacks = callbacks

    def _device_proto_to_device(self, device_proto):
        # TODO
        return None

    def RegisterWorker(self, request, context):
        # TODO(keshav2): Remove devices
        devices = []
        for device_proto in request.devices:
            devices.append(self._device_proto_to_device(device_proto))
        register_worker_callback = self._callbacks['RegisterWorker']
        try:
            worker_id, round_duration =\
                register_worker_callback(request.worker_type,
                                         request.ip_addr,
                                         request.port)
            return w2s_pb2.RegisterWorkerResponse(success=True,
                                                  worker_id=worker_id,
                                                  round_duration=round_duration)
        except Exception as e:
            # TODO: catch a more specific exception?
            print(e)
            return w2s_pb2.RegisterWorkerResponse(successful=False,
                                                  error_message=e)

    def SendHeartbeat(self, request, context):
        send_heartbeat_callback = self._callbacks['SendHeartbeat']
        send_heartbeat_callback()
        return common_pb2.Empty()

    def Done(self, request, context):
        done_callback = self._callbacks['Done']
        try:
            print('Received completion notification '
                  'from worker %d...' % (request.worker_id))
            if len(request.job_id) > 1:
                job_id = JobIdPair(request.job_id[0], request.job_id[1])
            else:
                job_id = JobIdPair(request.job_id[0], None)
            done_callback(job_id, request.worker_id,
                          request.num_steps, request.execution_time)
        except Exception as e:
            print(e)

        return common_pb2.Empty()

def serve(port, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
            SchedulerRpcServer(callbacks), server)
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    server.add_insecure_port('%s:%d' % (ip_address, port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
