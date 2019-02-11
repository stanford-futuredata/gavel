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
        self._worker_id = self.register_worker([])

    def _to_device_proto(self, device):
        return None

    def register_worker(self, devices):
        device_protos = [self.to_device_proto(device) for device in devices]
        request = w2s_pb2.RegisterWorkerRequest(devices=device_protos)
        with grpc.insecure_channel(self._server_loc) as channel:
            self._stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = self._stub.RegisterWorker(request)
            print(response.worker_id)
        return None

    def notify_scheduler(self, job_id):
        # Send a Done message.
        print("Trying to send notify scheduler to scheduler...")
        request = w2s_pb2.DoneRequest(job_id=job_id, worker_id=self._worker_id)
        response = self._stub.Done(request)
        print("Done and got response...")
        print('Notified scheduler of completion of %s on %s' % (job_id,
                                                                self._worker_id))
