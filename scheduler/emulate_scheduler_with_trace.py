import argparse
import datetime
import queue
import time
import datetime

import job
import policies
import scheduler
import utils

def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            job_type, command, num_steps_arg, total_steps, arrival_time, scale_factor = \
                    line.split('\t')
            jobs.append(job.Job(job_id=None,
                                job_type=job_type,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=int(total_steps),
                                duration=None,
                                scale_factor=int(scale_factor)))
            arrival_times.append(int(arrival_time))
    return jobs, arrival_times

def main(args):
    jobs, arrival_times = parse_trace(args.trace_file)
    policy = utils.get_policy(args.policy, args.seed)

    sched = scheduler.Scheduler(policy,
                                schedule_in_rounds=args.schedule_in_rounds,
                                throughputs_file=args.throughputs_file,
                                emulate=True,
                                seed=args.seed)

    cluster_spec = {key_value.split(':')[0]: int(key_value.split(':')[1])
                    for key_value in args.cluster_spec.split(',')}
    sched.emulate(cluster_spec, arrival_times, jobs,
                  ideal=args.ideal)
    sched.get_average_jct()
    sched.get_cluster_utilization()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-r', '--schedule_in_rounds', action='store_true',
                        help='Use rounds for scheduling')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=['max_min_fairness', 'max_min_fairness_perf',
                                 'max_min_fairness_packed',
                                 'min_total_duration',
                                 'min_total_duration_packed', 'fifo',
                                 'fifo_perf', 'fifo_packed'],
                        help='Scheduler policy')
    parser.add_argument('-i', '--ideal', action='store_true',
                        help='Use allocation returned by policy ideally')
    parser.add_argument('-f', '--throughputs_file', type=str,
                        default='combined_throughputs.json',
                        help='Throughputs file')
    parser.add_argument('-c', '--cluster_spec', type=str,
                        default='k80:4,p100:4,v100:4',
                        help='Cluster specification')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    main(parser.parse_args())
