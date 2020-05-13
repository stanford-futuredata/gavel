import grpc

import iterator_to_scheduler_pb2 as i2s_pb2
import iterator_to_scheduler_pb2_grpc as i2s_pb2_grpc

from lease import Lease

class IteratorRpcClient:

    def __init__(self, job_id, worker_id, sched_ip_addr, sched_port):
        self._job_id = job_id
        self._worker_id = worker_id
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)

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
                try:
                    response = stub.UpdateLease(request, timeout=TIMEOUT)
                    return (response.max_steps, response.max_duration)
                except grpc.RpcError as e:
                    attempts += 1
            return (0, 0)
