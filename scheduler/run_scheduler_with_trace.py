import argparse
import grpc
import numpy as np
import time

import job
import runtime.rpc.scheduler_client as scheduler_client
import scheduler

class TestPolicy:
    def get_allocation(self, throughputs):
        (m, n) = throughputs.shape
        return np.full((m, n), 1.0 / m)

def get_num_steps_to_run(job_id, worker_type):
    return 1

def read_trace(trace_filename):
    jobs = []
    # Trace file is expected to be in the following format:
    # <timestamp at which job is enqueued> <tab> <command> <tab> <duration> <tab> <number of times to run command>.
    with open(trace_filename, 'r') as f:
       for line in f.read().strip().split('\n'):
           [timestamp, command, duration, num_steps] = line.split('\t')
           job_id = None
           duration = float(duration)
           timestamp = int(timestamp)
           num_steps = int(num_steps)
           jobs.append((timestamp, job.Job(job_id,command, num_steps, duration)))
    jobs.sort(key=lambda x: x[0])
    return jobs

def main(trace_filename, min_workers, sleep_seconds, emulate):
    prev_timestamp = None
    s = scheduler.Scheduler(TestPolicy(), get_num_steps_to_run,
                            min_workers=min_workers, emulate=emulate)

    if emulate:
        for i in range(min_workers):
            s._register_worker_callback(
                worker_type="dummy_worker",
                ip_addr=None, port=None,
                devices=None)

    start = time.time()
    for (timestamp, job) in read_trace(trace_filename):
        if not emulate:
            if prev_timestamp is not None:
                time.sleep(timestamp - prev_timestamp)
            prev_timestamp = timestamp
        job_id = s.add_job(job)

    while s.num_jobs() > 0:
        time.sleep(sleep_seconds)

    if not emulate:
        print("Total time taken: %.2f seconds" % (time.time() - start))
    s.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute a trace"
    )
    parser.add_argument('-t', "--trace_filename", type=str, required=True,
                        help="Trace filename")
    parser.add_argument('-m', "--min_workers", type=int, default=None,
                        help="Minimum number of workers to wait for before " \
                             "scheduling jobs")
    parser.add_argument('-s', "--sleep_seconds", type=float, default=0.1,
                        help="Number of seconds to sleep when waiting for all" \
                             "jobs to complete")
    parser.add_argument('--emulate', action='store_true',
                        help="Emulate execution of jobs")
    args = parser.parse_args()

    main(args.trace_filename, args.min_workers, args.sleep_seconds,
         args.emulate)
