from concurrent import futures
import time

import grpc
import logging
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
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class SchedulerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self, callbacks, logger):
        self._callbacks = callbacks
        self._logger = logger

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
            self._logger.info(
                'Successfully registered {worker_type} worker '
                'with id(s) {worker_ids}'.format(
                    worker_type=request.worker_type,
                    worker_ids=str(worker_ids)))
            return w2s_pb2.RegisterWorkerResponse(success=True,
                                                  worker_ids=worker_ids,
                                                  round_duration=round_duration)
        except Exception as e:
            self._logger.error('Could not register worker: {0}'.format(e))
            return w2s_pb2.RegisterWorkerResponse(successful=False,
                                                  error_message=e)

    def SendHeartbeat(self, request, context):
        send_heartbeat_callback = self._callbacks['SendHeartbeat']
        send_heartbeat_callback()
        return common_pb2.Empty()

    def Done(self, request, context):
        done_callback = self._callbacks['Done']
        try:
            if len(request.job_id) > 1:
                job_id = JobIdPair(request.job_id[0], request.job_id[1])
            else:
                job_id = JobIdPair(request.job_id[0], None)
            self._logger.info(
                'Received completion notification: '
                'Job ID: {job_id}, Worker ID: {worker_id}, '
                'Num steps: {num_steps}, '
                'Execution time: {execution_time}'.format(
                    job_id=job_id, worker_id=request.worker_id,
                    num_steps=str(request.num_steps),
                    execution_time=str(request.execution_time)))
            done_callback(job_id, request.worker_id,
                          request.num_steps, request.execution_time)
        except Exception as e:
            self._logger.error('Could not process completion '
                               'notification: {0}'.format(e))

        return common_pb2.Empty()

class SchedulerIteratorRpcServer(i2s_pb2_grpc.IteratorToSchedulerServicer):
    def __init__(self, callbacks, logger):
        self._callbacks = callbacks
        self._logger = logger

    def InitJob(self, request, context):
        job_id = JobIdPair(request.job_id, None)
        self._logger.info(
            'Received job initialization request from job {0}'.format(job_id))
        init_job_callback = self._callbacks['InitJob']
        max_steps, max_duration, extra_time = init_job_callback(job_id=job_id)
        if max_steps > 0 and max_duration > 0:
            self._logger.info(
                'Initialized job {job_id} with initial lease '
                'max_steps={max_steps}, max_duration={max_duration:.2f}, '
                'extra_time={extra_time:.2f}'.format(
                    job_id=job_id, max_steps=max_steps,
                    max_duration=max_duration, extra_time=extra_time))
        else:
            self._logger.error('Failed to initialize job {0}!'.format(job_id))
        return i2s_pb2.UpdateLeaseResponse(max_steps=max_steps,
                                           max_duration=max_duration,
                                           extra_time=extra_time)

    def UpdateLease(self, request, context):
        job_id = JobIdPair(request.job_id, None)
        self._logger.info(
            'Received lease update request: '
            'job_id={job_id}, worker_id={worker_id}, steps={steps}, '
            'duration={duration:.2f}, max_steps={max_steps},'
            'max_duration={max_duration:.2f}'.format(
                job_id=job_id, worker_id=request.worker_id,
                steps=request.steps, duration=request.duration,
                max_steps=request.max_steps,
                max_duration=request.max_duration))

        update_lease_callback = self._callbacks['UpdateLease']
        try:
            (max_steps, max_duration) = \
                update_lease_callback(job_id=job_id,
                                      worker_id=request.worker_id,
                                      steps=request.steps,
                                      duration=request.duration,
                                      max_steps=request.max_steps,
                                      max_duration=request.max_duration)
            self._logger.info(
                'Sending new lease to job {job_id} (worker {worker_id}) '
                'with max_steps={max_steps}, '
                'max_duration={max_duration:.2f}'.format(
                    job_id=job_id, worker_id=request.worker_id,
                    max_steps=max_steps, max_duration=max_duration))
        except Exception as e:
            self._logger.error(
                'Could not update lease for job {0}: {1}'.format(
                    job_id, str(e)))
            max_steps = request.max_steps
            max_duration = request.max_duration

        return i2s_pb2.UpdateLeaseResponse(max_steps=max_steps,
                                           max_duration=max_duration)

def serve(port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor())
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
            SchedulerRpcServer(callbacks, logger), server)
    i2s_pb2_grpc.add_IteratorToSchedulerServicer_to_server(
            SchedulerIteratorRpcServer(callbacks, logger), server)
    ip_address = socket.gethostbyname(socket.gethostname())
    server.add_insecure_port('%s:%d' % (ip_address, port))
    logger.info('Starting server at {0}:{1}'.format(ip_address, port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
