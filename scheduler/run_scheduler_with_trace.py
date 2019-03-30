import argparse
import grpc
import time

import job
import policies
import runtime.rpc.scheduler_client as scheduler_client
import scheduler


def get_num_steps_to_run(job_id, worker_type):
    return 1

def read_trace(trace_filename):
    timestamps_and_jobs = []
    # Trace file is expected to be in the following format:
    # <timestamp at which job is enqueued> <tab> <job_type> <tab> <command> <tab> <duration> <tab> <number of times to run command>.
    with open(trace_filename, 'r') as f:
        for line in f.read().strip().split('\n'):
            print(line)
            [timestamp, job_type, command, duration, num_steps] = line.split('\t')
            job_id = None
            job_type = job_type
            duration = float(duration)
            timestamp = int(timestamp)
            num_steps = int(num_steps)
            timestamps_and_jobs.append(
                (timestamp,
                 job.Job(job_id, job_type, command, num_steps, duration)))
    timestamps_and_jobs.sort(key=lambda x: x[0])
    return timestamps_and_jobs

def main(trace_filename, policy_name, worker_types, num_workers,
         normalizing_worker_type, sleep_seconds, emulate,
         logfile, throughputs_directory):
    prev_timestamp = None
    policy = None
    if policy_name == "isolated":
        policy = policies.IsolatedPolicy()
    elif policy_name == "ks":
        policy = policies.KSPolicy()
    elif policy_name == "ks_packed":
        policy = policies.KSPolicyWithPacking()
    elif policy_name == "fifo":
        policy = policies.FIFOPolicy()
    elif policy_name == "max_throughput":
        policy = policies.MaximumThroughputPolicy()
    else:
        raise Exception("Unknown policy!")
    s = scheduler.Scheduler(policy, get_num_steps_to_run,
                            emulate=emulate,
                            normalizing_worker_type=normalizing_worker_type,
                            throughputs_directory=throughputs_directory,
                            job_packing=(policy_name == "ks_packed"))

    if emulate:
        for i in range(num_workers):
            worker_type = "dummy_worker"
            if worker_types is not None:
                worker_type = worker_types[i]
            s._register_worker_callback(
                worker_type=worker_type,
                ip_addr=None, port=None,
                devices=None, timestamp=0)

    start = time.time()
    for (timestamp, job) in read_trace(trace_filename):
        if not emulate:
            if prev_timestamp is not None:
                time.sleep(timestamp - prev_timestamp)
            prev_timestamp = timestamp
            job_id = s.add_job(job)
        else:
            s.add_to_event_queue(s.add_job, [job], timestamp)

    if emulate:
        s.start_scheduling_thread()

    while not s.is_done():
        time.sleep(sleep_seconds)

    if not emulate:
        print("Total time taken: %.2f seconds" % (time.time() - start))
    s.shutdown(logfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute a trace"
    )
    parser.add_argument('-t', "--trace_filename", type=str, required=True,
                        help="Trace filename")
    parser.add_argument("--policy_name", type=str, default="isolated",
                        help="Policy to use: fifo|isolated|ks|ks_packed|max_throughput")
    parser.add_argument('-w', "--worker_types", type=str, nargs='+',
                        help="Worker types: [k80|p100|v100]+")
    parser.add_argument('-n', "--num_workers", type=int, default=None,
                        help="Number of workers to use for scheduling jobs (in emulation mode)")
    parser.add_argument('-s', "--sleep_seconds", type=float, default=0.1,
                        help="Number of seconds to sleep when waiting for all" \
                             "jobs to complete")
    parser.add_argument('-l', "--logfile", type=str, default=None,
                        help="Log file to write final output to")
    parser.add_argument('--emulate', action='store_true',
                        help="Emulate execution of jobs")
    parser.add_argument("--throughputs_directory", type=str, default=None,
                        help="Directory with throughput measurements")
    args = parser.parse_args()

    args.normalizing_worker_type = None
    if args.worker_types is not None:
        assert args.num_workers is None, \
            "num_workers shouldn't be specified when worker_types is specified"
        args.num_workers = len(args.worker_types)
        args.normalizing_worker_type = args.worker_types[0]
    main(args.trace_filename, args.policy_name, args.worker_types,
         args.num_workers, args.normalizing_worker_type, args.sleep_seconds,
         args.emulate, args.logfile, args.throughputs_directory)
