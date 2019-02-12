import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc


class WorkerRpcClient:
    """Worker client for sending RPC requests to a scheduler server."""
  
    def __init__(self, server_ip_addr, port):
        self._server_loc = '%s:%d' % (server_ip_addr, port)
        #TODO: Get list of devices
        self._worker_id = self.register_worker([])

    def _to_device_proto(self, device):
        return None

    def register_worker(self, devices):
        device_protos = [self.to_device_proto(device) for device in devices]
        request = w2s_pb2.RegisterWorkerRequest(devices=device_protos)
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.RegisterWorker(request)
            print(response.worker_id)
        return None

    def send_heartbeat(self, jobs):
        #TODO
        pass

    def notify_scheduler(self, job_id):
        # Send a Done message.
        request = w2s_pb2.DoneRequest(job_id=job_id, worker_id=self._worker_id)
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.Done(request)
