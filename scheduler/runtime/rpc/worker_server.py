from concurrent import futures
import time

import grpc
import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
import enums_pb2

from .job import Job

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class WorkerServer(s2w_pb2_grpc.SchedulerToWorkerServicer):
    def __init__(self, callbacks, condition):
        self._callbacks = callbacks
        self._condition = condition

    def Run(self, request, context):
        jobs = []
        for job_description in request.job_descriptions:
            jobs.append(Job(job_description))
        run_callback = self._callbacks['Run']
        run_callback(jobs)
        return common_pb2.Empty()

    def Shutdown(self, request, context):
        # Handle any custom cleanup in the scheduler.
        shutdown_callback = self._callbacks['Shutdown']
        shutdown_callback()

        # Indiicate that the worker server has received a shutdown RPC.
        self._condition.acquire()
        self._condition.notify()
        self._condition.release()

        return common_pb2.Empty()

def serve(port, callbacks):
    condition = threading.Condition()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    s2w_pb2_grpc.add_SchedulerToWorkerServicer_to_server(
            WorkerServer(callbacks, condition), server)

    server.add_insecure_port('[::]:%d' % (port))
    server.start()

    # Wait for worker server to receive shutdown RPC from scheduler.
    condition.acquire()
    condition.wait()
    condition.release()
