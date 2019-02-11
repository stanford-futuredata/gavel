from concurrent import futures
import time

import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
import enums_pb2

import job

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class WorkerServer(s2w_pb2_grpc.SchedulerToWorkerServicer):
    def __init__(self, callbacks):
        self._callbacks = callbacks

    def Run(self, request, context):
        self.callbacks['Run'](job.Job(request))
        return common_pb2.Empty()

def serve(port, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    s2w_pb2_grpc.add_SchedulerToWorkerServicer_to_server(
            WorkerServer(callbacks), server)

    server.add_insecure_port('[::]:%d' % (port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
