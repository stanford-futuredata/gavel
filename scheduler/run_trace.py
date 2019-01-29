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

    def run_application(self, command, app_id, resource_id,
                        num_epochs):
        print("Running application_%d on %s for %d epochs: %s" %
            (app_id, resource_id, num_epochs, command))
        request = s2w_pb2.StartJobRequest(job_id=application_id,
                                          command=command)
        response = self.stub.StartJob(request)
        print("Job %d has status %s" % (response.job_id,
                                        enums_pb2.JobStatus.Name(response.status)))

def get_num_epochs_to_run(app_id, resource_id):
    return 1

def read_trace(trace_filename):
    commands = []
    with open(trace_filename, 'r') as f:
       for command in f.read():
           commands.append(command)
    return commands

def main(trace_filename):
    resource_ids = ["v100"]
    run_so_far = {}
    s = scheduler.Scheduler(resource_ids, TestPolicy(), GrpcStub(),
                            get_num_epochs_to_run)
    for command in read_trace(trace_filename):
        s.add_new_job({resource_id: 10 for resource_id in resource_ids},
                      command)
    for i in range(100):  # Some arbitrary number.
        resource_id = s.get_available_resource()
        app_id, num_epochs = s.schedule(resource_id)
        # Leverage the fact that there's only one resource right now.
        # while s.get_status(app_id) == RUNNING:  # Busy wait till app is done.
        #     pass
        s.schedule_callback(app_id, resource_ids[0], num_epochs)
        if app_id not in run_so_far:
            run_so_far[app_id] = 0
        run_so_far[app_id] += num_epochs
    print()
    print("Number of epochs run:", run_so_far)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute a trace"
    )
    parser.add_argument('-t', "--trace_filename", type=str, required=True,
                        help="Trace filename")
    args = parser.parse_args()

    main(args.trace_filename)
