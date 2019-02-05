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
    commands_and_num_epochs = []
    with open(trace_filename, 'r') as f:
       for command_and_num_epochs in f.read().strip().split('\n'):
           [command, num_epochs] = command_and_num_epochs.split('\t')
           num_epochs = int(num_epochs)
           commands_and_num_epochs.append((command, num_epochs))
    return commands_and_num_epochs

def main(trace_filename):
    worker_ids = [1]
    for worker_id in worker_ids:
        scheduler_client.register_worker(worker_id)
    num_epochs_left = {}
    s = scheduler.Scheduler(worker_ids, TestPolicy(), scheduler_client.run,
                            get_num_epochs_to_run, run_server=True)
    for (command, num_epochs) in read_trace(trace_filename):
        job_id = s.add_new_job({worker_id: 10 for worker_id in worker_ids},
                               command)
        num_epochs_left[job_id] = num_epochs

    import time
    start = time.time()
    job_id, worker_id, num_epochs = None, None, None
    while len(num_epochs_left) > 0:
        old_job_id, old_worker_id, old_num_epochs = \
            job_id, worker_id, num_epochs
        job_id, worker_id, num_epochs = s._schedule()
        if old_job_id in num_epochs_left:
            num_epochs_left[old_job_id] -= old_num_epochs
            if num_epochs_left[old_job_id] == 0:
                del num_epochs_left[old_job_id]
                s.remove_old_job(old_job_id)
        if job_id is None:
            break
        print("Number of epochs left:", num_epochs_left)
    print("Total time taken: %.2f seconds" % (time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute a trace"
    )
    parser.add_argument('-t', "--trace_filename", type=str, required=True,
                        help="Trace filename")
    args = parser.parse_args()

    main(args.trace_filename)
