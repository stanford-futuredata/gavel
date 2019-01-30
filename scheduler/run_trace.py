import argparse
import grpc
import numpy as np

import runtime.rpc.scheduler_client as scheduler_client
import scheduler

class TestPolicy:
    def get_allocation(self, throughputs):
        (m, n) = throughputs.shape
        return np.full((m, n), 1.0 / n)

def get_num_epochs_to_run(job_id, worker_id):
    return 1

def read_trace(trace_filename):
    commands = []
    with open(trace_filename, 'r') as f:
       for command in f.read().strip().split('\n'):
           commands.append(command)
    return commands

def main(trace_filename):
    worker_ids = [1]
    run_so_far = {}
    s = scheduler.Scheduler(worker_ids, TestPolicy(), scheduler_client.run,
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
