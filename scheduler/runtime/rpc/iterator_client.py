import grpc

import iterator_to_scheduler_pb2 as i2s_pb2
import iterator_to_scheduler_pb2_grpc as i2s_pb2_grpc

from lease import Lease

class IteratorRpcClient:

    def __init__(self, job_id, worker_id, sched_ip_addr, sched_port, logger):
        self._job_id = job_id
        self._worker_id = worker_id
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)
        self._logger = logger

    def init(self):
        request = i2s_pb2.InitJobRequest(job_id=self._job_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = i2s_pb2_grpc.IteratorToSchedulerStub(channel)
            try:
                self._logger.info(
                    '', extra={'event': 'INIT', 'status': 'REQUESTING'})
                response = stub.InitJob(request)
                if response.max_steps > 0 and response.max_duration > 0:
                    self._logger.info(
                        'Initial lease: max_steps {0}, '
                        'max_duration={1:.4f}'.format(
                            response.max_steps, response.max_duration),
                        extra={'event': 'INIT', 'status': 'COMPLETE'})
                else:
                    self._logger.error(
                        '', extra={'event': 'INIT', 'status': 'FAILED'})
                return (response.max_steps, response.max_duration,
                        response.extra_time)
            except grpc.RpcError as e:
                self._logger.error(
                    '{0}'.format(e),
                    extra={'event': 'INIT', 'status': 'ERROR'})
            return (0, 0, 0)

    def update_lease(self, steps, duration, max_steps, max_duration):
        request = i2s_pb2.UpdateLeaseRequest(job_id=self._job_id,
                                             worker_id=self._worker_id,
                                             steps=steps,
                                             duration=duration,
                                             max_steps=max_steps,
                                             max_duration=max_duration)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = i2s_pb2_grpc.IteratorToSchedulerStub(channel)
            self._logger.info(
                '', extra={'event': 'LEASE', 'status': 'REQUESTING'})
            try:
                response = stub.UpdateLease(request)
                self._logger.info(
                    'New lease: max_steps={0}, max_duration={1:.4f}'.format(
                        response.max_steps, response.max_duration),
                          extra={'event': 'LEASE', 'status': 'UPDATED'})
                return (response.max_steps, response.max_duration)
            except grpc.RpcError as e:
                self._logger.error(
                    '{0}'.format(e),
                    extra={'event': 'LEASE', 'status': 'ERROR'})
        return (max_steps, max_duration)
