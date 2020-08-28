import grpc

import iterator_to_scheduler_pb2 as i2s_pb2
import iterator_to_scheduler_pb2_grpc as i2s_pb2_grpc

from lease import Lease

MAX_ATTEMPTS = 5
TIMEOUT = 5

class IteratorRpcClient:

    def __init__(self, job_id, worker_id, sched_ip_addr, sched_port,
                 verbose=True):
        self._job_id = job_id
        self._worker_id = worker_id
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)
        self._verbose = verbose

    def _log(self, message):
        if self._verbose:
            print('[Job %d | worker %d] %s' % (self._job_id,
                                               self._worker_id,
                                               message))

    def init(self):
        request = i2s_pb2.InitJobRequest(job_id=self._job_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = i2s_pb2_grpc.IteratorToSchedulerStub(channel)
            try:
                response = stub.InitJob(request)
                if response.max_steps > 0 and response.max_duration > 0:
                    self._log(
                        'Initialized job %d with initial lease max_steps=%d, '
                        'max_duration=%f' % (self._job_id, response.max_steps,
                                             response.max_duration))
                else:
                    self._log('Failed to initialize job %d!' % (self._job_id))
                return (response.max_steps, response.max_duration)
            except grpc.RpcError as e:
                self._log('Job initialization error: %s' % (e))
            return(0, 0)

    def update_lease(self, steps, duration, max_steps, max_duration):
        request = i2s_pb2.UpdateLeaseRequest(job_id=self._job_id,
                                             worker_id=self._worker_id,
                                             steps=steps,
                                             duration=duration,
                                             max_steps=max_steps,
                                             max_duration=max_duration)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = i2s_pb2_grpc.IteratorToSchedulerStub(channel)
            attempts = 0
            while attempts < MAX_ATTEMPTS:
                self._log('Requesting new lease '
                          '(attempt %d)...' % (attempts + 1))
                try:
                    response = stub.UpdateLease(request, timeout=TIMEOUT)
                    self._log('Received new lease: '
                              '%d, %f' % (response.max_steps,
                                          response.max_duration))
                    return (response.max_steps, response.max_duration)
                except grpc.RpcError as e:
                    self._log('UpdateLease error: %s' % (e))
                    attempts += 1
            self._log('WARNING: Failed to receive new lease!')
            return (max_steps, max_duration)
