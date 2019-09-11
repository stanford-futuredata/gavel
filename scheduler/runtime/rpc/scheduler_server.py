from concurrent import futures
import time

import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

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
            worker_id = register_worker_callback(request.worker_type,
                                                 request.ip_addr,
                                                 request.port)
            return w2s_pb2.RegisterWorkerResponse(worker_id=worker_id)
        except Exception as e:
            # TODO: catch a more specific exception?
            print(e)
            return w2s_pb2.RegisterWorkerResponse(error_message=e)

    def SendHeartbeat(self, request, context):
        send_heartbeat_callback = self._callbacks['SendHeartbeat']
        send_heartbeat_callback()
        return common_pb2.Empty()

    def Done(self, request, context):
        done_callback = self._callbacks['Done']
        try:
            worker_id = None
            all_job_ids = []
            all_num_steps = []
            all_execution_times = []
            for job_description in request.job_descriptions:
                if worker_id is None:
                    worker_id = job_description.worker_id
                else:
                    assert(worker_id == job_description.worker_id)
                all_job_ids.append(job_description.job_id)
                all_num_steps.append(job_description.num_steps)
                all_execution_times.append(job_description.execution_time)
            if len(all_job_ids) == 1:
                all_job_ids.append(None)
            done_callback(JobIdPair(job0=all_job_ids[0], job1=all_job_ids[1]),
                          worker_id, all_num_steps, all_execution_times)
        except Exception as e:
            print(e)
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
