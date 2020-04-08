from concurrent import futures
import time

import grpc
import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
import enums_pb2

import job

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class WorkerServer(s2w_pb2_grpc.SchedulerToWorkerServicer):
    def __init__(self, callbacks, condition, write_queue):
        self._callbacks = callbacks
        self._condition = condition
        self._write_queue = write_queue

    def Run(self, request, context):
        jobs = []
        for job_description in request.job_descriptions:
            jobs.append(job.Job.from_proto(job_description))
        run_callback = self._callbacks['Run']
        run_callback(jobs, request.worker_id, request.send_output)
        return common_pb2.Empty()

    def Shutdown(self, request, context):
        # Handle any custom cleanup in the scheduler.
        shutdown_callback = self._callbacks['Shutdown']
        shutdown_callback()

        # Indicate to the worker server that a shutdown RPC has been received.
        self._condition.acquire()
        self._condition.notify()
        self._condition.release()

        return common_pb2.Empty()

def serve(port, callbacks, write_queue):
    condition = threading.Condition()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    s2w_pb2_grpc.add_SchedulerToWorkerServicer_to_server(
            WorkerServer(callbacks, condition, write_queue), server)

    write_queue.put('Starting server at port %s' % (str(port)))
    server.add_insecure_port('[::]:%d' % (port))
    server.start()

    # Wait for worker server to receive a shutdown RPC from scheduler.
    condition.acquire()
    condition.wait()
    condition.release()

    # Wait for shutdown message to be sent to scheduler.
    time.sleep(5)
