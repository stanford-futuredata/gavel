import grpc
import logging
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc

MAX_ATTEMPTS = 5
SLEEP_SECONDS = 5
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class WorkerRpcClient:
    """Worker client for sending RPC requests to a scheduler server."""

    def __init__(self, worker_type, worker_ip_addr, worker_port,
                 sched_ip_addr, sched_port):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._worker_type = worker_type
        self._worker_ip_addr = worker_ip_addr
        self._worker_port = worker_port
        self._sched_ip_addr = sched_ip_addr
        self._sched_port = sched_port
        # TODO: Remove self._sched_ip_addr and self._sched_port?
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)

    def register_worker(self, num_gpus):
        request = w2s_pb2.RegisterWorkerRequest(
            worker_type=self._worker_type,
            ip_addr=self._worker_ip_addr,
            port=self._worker_port,
            num_gpus=num_gpus)
        with grpc.insecure_channel(self._sched_loc) as channel:
            self._logger.debug('Trying to register worker...')
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.RegisterWorker(request)
            if response.success:
                self._logger.info(
                    'Succesfully registered worker with id(s) {worker_id}, '
                    'round_duration={round_duration}'.format(
                        worker_id=str(response.worker_ids),
                        round_duration=response.round_duration))
                return (response.worker_ids, response.round_duration, None)
            else:
                assert(response.HasField('error'))
                self._logger.error('Failed to register worker!')
                return (None, response.error)

    def send_heartbeat(self, jobs):
        #TODO
        pass

    def notify_scheduler(self, worker_id, job_descriptions):
        # Send a Done message.
        request = w2s_pb2.DoneRequest()
        request.worker_id = worker_id
        for (job_id, execution_time, num_steps) in job_descriptions:
            request.job_id.append(job_id)
            request.execution_time.append(execution_time)
            request.num_steps.append(num_steps)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.Done(request)
            job_ids = \
              [job_description[0] for job_description in job_descriptions]
            if len(job_ids) == 1:
              self._logger.debug('Notified scheduler that '
                                 'job {0} has completed'.format(str(job_ids)))
