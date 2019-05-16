import argparse
import datetime
import queue
import time
import datetime

import job
import policies
import emulator

def get_policy(policy_name):
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
    return policy

def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            job_type, command, num_steps_arg, total_steps, arrival_time = \
                    line.split('\t')
            jobs.append(job.Job(job_id=None,
                                job_type=job_type,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=int(total_steps),
                                duration=None))
            arrival_times.append(int(arrival_time))
    return jobs, arrival_times

def main(args):
    jobs, arrival_times = parse_trace(args.trace_file)
    policy = get_policy(args.policy)
    sched = emulator.Scheduler(policy, args.throughputs_file,
                               job_packing=False)
    start_time = datetime.datetime.now()
    cluster_spec = {
          'k80': 4,
          'p100': 4,
          'v100': 4,
    }
    sched.emulate(cluster_spec, arrival_times, jobs)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=['isolated', 'ks', 'ks_packed', 'fifo',
                                 'max_throughput'],
                        help='Scheduler policy')
    parser.add_argument('-r', '--throughputs_file', type=str,
                        default='throughputs.json',
                        help='Throughputs file')
    main(parser.parse_args())
