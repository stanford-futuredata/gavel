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
import iterator_to_scheduler_pb2 as i2s_pb2
import iterator_to_scheduler_pb2_grpc as i2s_pb2_grpc
import common_pb2
from job_id_pair import JobIdPair

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class SchedulerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self, callbacks, write_queue):
        self._callbacks = callbacks
        self._write_queue = write_queue

    def _device_proto_to_device(self, device_proto):
        # TODO
        return None

    def RegisterWorker(self, request, context):
        register_worker_callback = self._callbacks['RegisterWorker']
        try:
            worker_ids, round_duration =\
                register_worker_callback(worker_type=request.worker_type,
                                         num_gpus=request.num_gpus,
                                         ip_addr=request.ip_addr,
                                         port=request.port)
            self._write_queue.put('Successfully registered %s worker '
                                  'with id(s) %s' % (request.worker_type,
                                                     str(worker_ids)))
            return w2s_pb2.RegisterWorkerResponse(success=True,
                                                  worker_ids=worker_ids,
                                                  round_duration=round_duration)
        except Exception as e:
            self._write_queue.put('Could not register worker: %s' % (e))
            return w2s_pb2.RegisterWorkerResponse(successful=False,
                                                  error_message=e)

    def SendHeartbeat(self, request, context):
        send_heartbeat_callback = self._callbacks['SendHeartbeat']
        send_heartbeat_callback()
        return common_pb2.Empty()

    def Done(self, request, context):
        done_callback = self._callbacks['Done']
        self._write_queue.put(
            'Received completion notification '
            'from worker %d...' % (request.worker_id))
        try:
            if len(request.job_id) > 1:
                job_id = JobIdPair(request.job_id[0], request.job_id[1])
            else:
                job_id = JobIdPair(request.job_id[0], None)
            done_callback(job_id, request.worker_id,
                          request.num_steps, request.execution_time)
        except Exception as e:
            self._write_queue.put('Could not process completion '
                                  'notification: %s' % (e))

        return common_pb2.Empty()

class SchedulerIteratorRpcServer(i2s_pb2_grpc.IteratorToSchedulerServicer):
    def __init__(self, callbacks, write_queue):
        self._callbacks = callbacks
        self._write_queue = write_queue

    def InitJob(self, request, context):
        job_id = request.job_id
        self._write_queue.put(
            'Received job initialization request from job %d' % (job_id))
        init_job_callback = self._callbacks['InitJob']
        init_job_callback(job_id=job_id)
        self._write_queue.put('Initialized job %d' % (job_id))
        return common_pb2.Empty()

    def UpdateLease(self, request, context):
        self._write_queue.put('Received lease update request: '
                              'job_id=%s, '
                              'worker_id=%d, '
                              'steps=%d, '
                              'duration=%f, '
                              'max_steps=%d,'
                              'max_duration=%f' % (request.job_id,
                                                   request.worker_id,
                                                   request.steps,
                                                   request.duration,
                                                   request.max_steps,
                                                   request.max_duration))

        update_lease_callback = self._callbacks['UpdateLease']
        (max_steps, max_duration) = \
            update_lease_callback(job_id=JobIdPair(request.job_id, None),
                                  worker_id=request.worker_id,
                                  steps=request.steps,
                                  duration=request.duration,
                                  max_steps=request.max_steps,
                                  max_duration=request.max_duration)
        self._write_queue.put('Sending new lease to job %d (worker %d) '
                              'with max_steps=%d, '
                              'max_duration=%f' % (request.job_id,
                                                   request.worker_id,
                                                   max_steps,
                                                   max_duration))

        return i2s_pb2.UpdateLeaseResponse(max_steps=max_steps,
                                           max_duration=max_duration)

def serve(port, callbacks, write_queue):
    server = grpc.server(futures.ThreadPoolExecutor())
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
            SchedulerRpcServer(callbacks, write_queue), server)
    i2s_pb2_grpc.add_IteratorToSchedulerServicer_to_server(
            SchedulerIteratorRpcServer(callbacks, write_queue), server)
    ip_address = socket.gethostbyname(socket.gethostname())
    server.add_insecure_port('%s:%d' % (ip_address, port))
    write_queue.put('Starting server at %s:%s' % (ip_address, port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
