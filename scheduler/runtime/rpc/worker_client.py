import grpc
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc

MAX_ATTEMPTS = 5
SLEEP_SECONDS = 5

class WorkerRpcClient:
    """Worker client for sending RPC requests to a scheduler server."""
  
    def __init__(self, worker_ip_addr, worker_port, sched_ip_addr, sched_port):
        self._worker_ip_addr = worker_ip_addr
        self._worker_port = worker_port
        self._sched_ip_addr = sched_ip_addr
        self._sched_port = sched_port
        # TODO: Remove self._sched_ip_addr and self._sched_port?
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)
        #TODO: Get list of devices

    def _to_device_proto(self, device):
        return None

    def register_worker(self, devices):
        attempts = 0
        device_protos = [self.to_device_proto(device) for device in devices]
        request = w2s_pb2.RegisterWorkerRequest(
                    ip_addr=self._worker_ip_addr,
                    port=self._worker_port,
                    devices=device_protos)
        with grpc.insecure_channel(self._sched_loc) as channel:
            while attempts < MAX_ATTEMPTS:
                stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
                response = stub.RegisterWorker(request)
                if response.HasField('worker_id'):
                    return (response.worker_id, None)
                elif attempts == MAX_ATTEMPTS:
                    assert(response.HasField('error'))
                    return (None, response.error)
                else:
                    attempts += 1
                    time.sleep(SLEEP_SECONDS)

    def send_heartbeat(self, jobs):
        #TODO
        pass

    def notify_scheduler(self, job_id, worker_id):
        # Send a Done message.
        request = w2s_pb2.DoneRequest(job_id=job_id, worker_id=worker_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.Done(request)
