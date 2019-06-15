import argparse
import datetime
import queue
import time
import datetime

import job
import policies
import scheduler

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

def sweep_cluster_sizes(jobs, arrival_times, policy, schedule_in_rounds,
                        throughputs_file, output_file):
    ratios = [(1, 0, 0), (1, 1, 1), (1, 1, 0), (2, 1, 0)]
    with open(output_file, 'w') as f:
        f.write('# v100\t# p100\t# k80\tUtilization\tAverage JCT\n')
        f.flush()
        for ratio in ratios:
            for i in range(8):
                cluster_spec = {}
                cluster_spec['v100'] = ratio[0] * (2 ** i)
                cluster_spec['p100'] = ratio[1] * (2 ** i)
                cluster_spec['k80'] = ratio[2] * (2 ** i)
                sched = \
                    scheduler.Scheduler(policy,
                                        schedule_in_rounds=schedule_in_rounds,
                                        throughputs_file=throughputs_file,
                                        emulate=True)
                sched.emulate(cluster_spec, arrival_times, jobs, ideal=False)
                utilization = sched.get_cluster_utilization()
                if utilization is None:
                    continue
                average_jct = sched.shutdown()
                f.write('%d\t%d\t%d\t%.3f\t%.3f\n' % (cluster_spec['v100'],
                                                      cluster_spec['p100'],
                                                      cluster_spec['k80'],
                                                      utilization,
                                                      average_jct))
                f.flush()

def main(args):
    jobs, arrival_times = parse_trace(args.trace_file)
    policy = get_policy(args.policy)

    # TODO: Sweep based on command line arguments.
    # for policy_name in ['ks_packed', 'ks']:
    #    output_file = '%s_utilization.csv' % (policy_name)
    #    sweep_cluster_sizes(jobs, arrival_times, policy,
    #                        args.schedule_in_rounds,
    #                        args.throughputs_file, output_file)

    sched = scheduler.Scheduler(policy,
                                schedule_in_rounds=args.schedule_in_rounds,
                                throughputs_file=args.throughputs_file,
                                emulate=True)

    cluster_spec = {key_value.split(':')[0]: int(key_value.split(':')[1])
                    for key_value in args.cluster_spec.split(',')}
    sched.emulate(cluster_spec, arrival_times, jobs, ideal=args.ideal)
    sched.get_average_jct()
    sched.get_cluster_utilization()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-r', '--schedule_in_rounds', action='store_true',
                        help='Use rounds for scheduling')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=['isolated', 'ks', 'ks_packed', 'fifo',
                                 'max_throughput'],
                        help='Scheduler policy')
    parser.add_argument('-i', '--ideal', action='store_true',
                        help='Use allocation returned by policy ideally')
    parser.add_argument('-f', '--throughputs_file', type=str,
                        default='combined_throughputs.json',
                        help='Throughputs file')
    parser.add_argument('-c', '--cluster_spec', type=str,
                        default='k80:4,p100:4,v100:4',
                        help='Cluster specification')

    main(parser.parse_args())
