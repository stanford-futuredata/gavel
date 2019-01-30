import argparse
import grpc
import numpy as np

import scheduler
import grpc_stubs.scheduler_to_worker_pb2 as s2w_pb2
import grpc_stubs.scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import grpc_stubs.enums_pb2

class TestPolicy:
    def get_allocation(self, throughputs):
        (m, n) = throughputs.shape
        return np.full((m, n), 1.0 / n)

class GrpcStub:
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:50052')
        self.stub = s2w_pb2_grpc.SchedulerToWorkerStub(self.channel)

    def run_application(self, command, job_id, worker_id,
                        num_epochs):
        print("Running application_%d on %s for %d epochs: %s" %
            (job_id, worker_id, num_epochs, command))
        request = s2w_pb2.StartJobRequest(job_id=application_id,
                                          command=command)
        response = self.stub.StartJob(request)
        print("Job %d has status %s" % (response.job_id,
                                        enums_pb2.JobStatus.Name(response.status)))

def get_num_epochs_to_run(job_id, worker_id):
    return 1

def read_trace(trace_filename):
    commands = []
    with open(trace_filename, 'r') as f:
       for command in f.read():
           commands.append(command)
    return commands

def main(trace_filename):
    worker_ids = ["worker1"]
    run_so_far = {}
    s = scheduler.Scheduler(worker_ids, TestPolicy(), GrpcStub(),
                            get_num_epochs_to_run, run_server=True)
    for command in read_trace(trace_filename):
        s.add_new_job({worker_id: 10 for worker_id in worker_ids},
                      command)
    for i in range(100):  # Some arbitrary number.
        job_id, worker_id, num_epochs = s._schedule()
        # TODO: Do something here.
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute a trace"
    )
    parser.add_argument('-t', "--trace_filename", type=str, required=True,
                        help="Trace filename")
    args = parser.parse_args()

    main(args.trace_filename)
