import argparse
import datetime
import json
import contextlib
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing
import numpy as np
import pickle
import os
import sys

from job_id_pair import JobIdPair
import scheduler
import utils


def simulate_with_timeout(experiment_id, policy_name,
                          throughputs_file, cluster_spec, interval,
                          seed, vc, log_dir, trace_dir, run_dir, timeout,
                          verbose):
    with open('philly_steady_state_jobs', 'rb') as f:
        philly_steady_state_jobs = pickle.load(f)

    with open('philly_jobs_to_complete', 'rb') as f:
        jobs_to_complete = pickle.load(f)
    jobs_to_complete = list(jobs_to_complete)

    for i in range(len(jobs_to_complete)):
        jobs_to_complete[i] = JobIdPair(jobs_to_complete[i], None)
    jobs_to_complete = set(jobs_to_complete)

    input_trace = os.path.join(trace_dir, 'seed=%d' % seed,
                               '%s.trace' % (vc))
    output_file = 'seed=%d.log' % seed
    jobs, arrival_times = utils.parse_trace(input_trace, run_dir)
    pruned_jobs, pruned_arrival_times = [], []
    for job_id, (job, arrival_time) in enumerate(zip(jobs, arrival_times)):
        if job_id in philly_steady_state_jobs:
            pruned_jobs.append(job)
            pruned_arrival_times.append(arrival_time)
    jobs = pruned_jobs
    arrival_times = pruned_arrival_times

    with open(os.path.join(log_dir, output_file), 'w') as f:
        with contextlib.redirect_stdout(f):
            policy = utils.get_policy(policy_name, seed)
            sched = scheduler.Scheduler(
                            policy,
                            throughputs_file=throughputs_file,
                            seed=seed,
                            time_per_iteration=interval,
                            simulate=True)

            cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                           cluster_spec['p100'],
                                                           cluster_spec['k80'])
            if verbose:
                current_time = datetime.datetime.now()
                print('[%s] [Experiment ID: %2d] '
                      'Configuration: cluster_spec=%s, policy=%s, '
                       'seed=%d, vc=%s' % (current_time, experiment_id,
                                           cluster_spec_str, policy.name,
                                           seed, vc),
                      file=sys.stderr)

            if timeout is None:
                sched.simulate(cluster_spec, arrival_times=arrival_times,
                               jobs=jobs,
                               jobs_to_complete=jobs_to_complete)
                average_jct = sched.get_average_jct()
                utilization = sched.get_cluster_utilization()
            else:
                try:
                    func_timeout(timeout, sched.simulate,
                                 args=(cluster_spec,),
                                 kwargs={
                                    'arrival_times': arrival_times,
                                    'jobs': jobs,
                                    'jobs_to_complete': jobs_to_complete
                                })
                    average_jct = sched.get_average_jct()
                    utilization = sched.get_cluster_utilization()
                except FunctionTimedOut:
                    average_jct = float('inf')
                    utilization = 1.0

    if verbose:
        current_time = datetime.datetime.now()
        print('[%s] [Experiment ID: %2d] '
              'Results: average JCT=%f, utilization=%f' % (current_time,
                                                           experiment_id,
                                                           average_jct,
                                                           utilization),
              file=sys.stderr)

    return average_jct, utilization

def sweep_cluster_sizes_helper(experiment_id, throughputs_file, interval, vc,
                               seed, log_dir, trace_dir, run_dir,
                               util_threshold):
    input_trace = os.path.join(trace_dir, 'seed=%d' % seed,
                               '%s.trace' % (vc))
    jobs, arrival_times = utils.parse_trace(input_trace, run_dir)
    num_v100s = 1024
    i = 0
    while True:
        cluster_spec = {
            'v100': num_v100s,
        }
        log_file = 'v100=%d.log' % (num_v100s)
        with open(os.path.join(log_dir, log_file), 'w') as f:
            with contextlib.redirect_stdout(f):
                policy = utils.get_policy('fifo', seed=seed)
                sched = scheduler.Scheduler(
                            policy,
                            throughputs_file=throughputs_file,
                            seed=seed,
                            time_per_iteration=interval,
                            simulate=True)
                current_time = datetime.datetime.now()
                print('[%s] [Experiment ID: %2d.%d] '
                      'Configuration: v100=%d, '
                      'seed=%d, vc=%s' % (current_time, experiment_id, i,
                                          num_v100s, seed, vc),
                      file=sys.stderr)
                sched.simulate(cluster_spec, arrival_times=arrival_times,
                               jobs=jobs)
                average_jct = sched.get_average_jct()
                utilization = sched.get_cluster_utilization()
                current_time = datetime.datetime.now()
                print('[%s] [Experiment ID: %2d.%d] '
                      'Results: average JCT=%f, '
                      'utilization=%f' % (current_time, experiment_id, i,
                                          average_jct, utilization),
                      file=sys.stderr)
                if utilization >= util_threshold or num_v100s == 1:
                    break
                else:
                    i += 1
                    num_v100s = int(num_v100s / 2)
    return (vc, seed, num_v100s)

def sweep_cluster_sizes(args):
    all_args_list = []
    experiment_id = 0

    cluster_size_log_dir = os.path.join(args.log_dir, 'cluster_sizes')
    if not os.path.isdir(cluster_size_log_dir):
        os.mkdir(cluster_size_log_dir)

    for vc in args.vcs:
        vc_dir = os.path.join(cluster_size_log_dir, 'vc=%s' % (vc))
        if not os.path.isdir(vc_dir):
            os.mkdir(vc_dir)
        for seed in args.seeds:
            seed_dir = os.path.join(vc_dir, 'seed=%d' % (seed))
            if not os.path.isdir(seed_dir):
                os.mkdir(seed_dir)
            all_args_list.append((experiment_id, args.throughputs_file,
                                  args.interval, vc, seed, seed_dir,
                                  args.trace_dir, args.run_dir,
                                  args.util_threshold))
            experiment_id += 1

    if len(all_args_list) > 0:
        current_time = datetime.datetime.now()
        print('[%s] Running %d total experiment(s)...' % (current_time,
                                                          len(all_args_list)))
        with multiprocessing.Pool(args.processes) as p:
            # TODO: Sort all_args_list?
            results = [p.apply_async(sweep_cluster_sizes_helper, args_list)
                       for args_list in all_args_list]
            results = [result.get() for result in results]
        cluster_sizes = {}
        for (vc, seed, v100) in results:
            if vc not in cluster_sizes:
                cluster_sizes[vc] = {}
            cluster_sizes[vc][int(seed)] = int(v100)
        output_file = os.path.join(cluster_size_log_dir,
                                   args.cluster_size_output_file)
        with open(output_file, 'w') as f:
            f.write(json.dumps(cluster_sizes, indent=4))
    else:
        raise ValueError('No work to be done!')

def sweep_policies(args):
    experiment_id = 0
    throughputs_file = args.throughputs_file
    policy_names = args.policies

    policy_log_dir = os.path.join(args.log_dir, 'policies')
    if not os.path.isdir(policy_log_dir):
        os.mkdir(policy_log_dir)

    vc_and_cluster_specs = json.load(open(args.cluster_spec_file, 'r'))

    all_args_list = []
    for vc, cluster_spec_strs in vc_and_cluster_specs.items():
        for cluster_spec_str in cluster_spec_strs:

            cluster_spec_str_split = cluster_spec_str.split(':')
            if len(cluster_spec_str_split) != 3:
                raise ValueError('Invalid cluster spec %s' % (cluster_spec_str))
            cluster_spec = {
                'v100': int(cluster_spec_str_split[0]),
                'p100': int(cluster_spec_str_split[1]),
                'k80': int(cluster_spec_str_split[2]),
            }

            vc_subdir = os.path.join(policy_log_dir, 'vc=%s' % vc)
            if not os.path.isdir(vc_subdir):
                os.mkdir(vc_subdir)

            cluster_spec_subdir = \
                os.path.join(vc_subdir,
                             'v100=%d.p100=%d.k80=%d' % (cluster_spec['v100'],
                                                         cluster_spec['p100'],
                                                         cluster_spec['k80']))
            if not os.path.isdir(cluster_spec_subdir):
                os.mkdir(cluster_spec_subdir)

            for policy_name in policy_names:
                policy_subdir = os.path.join(cluster_spec_subdir,
                                             policy_name)
                if not os.path.isdir(policy_subdir):
                    os.mkdir(policy_subdir)

                for seed in args.seeds:
                    all_args_list.append((experiment_id, policy_name,
                                          throughputs_file, cluster_spec,
                                          args.interval, seed, vc, policy_subdir,
                                          args.trace_dir, args.run_dir,
                                          args.timeout, args.verbose))
                    experiment_id += 1

    if len(all_args_list) > 0:
        current_time = datetime.datetime.now()
        print('[%s] Running %d total experiment(s)...' % (current_time,
                                                          len(all_args_list)))
        with multiprocessing.Pool(args.processes) as p:
            # TODO: Sort all_args_list?
            results = [p.apply_async(simulate_with_timeout, args_list)
                       for args_list in all_args_list]
            results = [result.get() for result in results]
    else:
        raise ValueError('No work to be done!')


def main(args):
    if args.sweep_cluster_sizes:
        sweep_cluster_sizes(args)
    else:
        sweep_policies(args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Sweep through Philly traces')
    cluster_size_args = parser.add_argument_group('Sweep cluster sizes')
    policy_args = parser.add_argument_group('Sweep policies')

    parser.add_argument('-l', '--log-dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('-j', '--processes', type=int, default=None,
                        help=('Number of processes to use in pool '
                              '(use as many as available if not specified)'))
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4],
                        help='List of random seeds')
    parser.add_argument('--trace_dir', type=str, default='traces/msr/',
                        help='Path to Philly traces')
    parser.add_argument('--vcs', type=str, nargs='+',
                        help='Virtual clusters (VCs)')
    parser.add_argument('-i', '--interval', type=int, default=1920,
                        help='Interval length (in seconds)')
    parser.add_argument('--throughputs-file', type=str,
                        default='oracle_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('--run_dir', type=str, default='/tmp',
                        help='Run directory')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose')
    policy_args.add_argument('--timeout', type=int, default=None,
                             help='Timeout (in seconds) for each run')
    policy_args.add_argument('-p', '--policies', type=str, nargs='+',
                             default=['fifo', 'fifo_perf', 'fifo_packed',
                                      'max_min_fairness',
                                      'max_min_fairness_perf',
                                      'max_min_fairness_packed'],
                             help='List of policies to sweep')
    parser.add_argument('-c', '--cluster-spec-file', type=str, required=True,
                        help=('Cluster specification file'))
    cluster_size_args.add_argument('--sweep-cluster-sizes',
                                   action='store_true',
                                   default=False,
                                   help=('Search for the right '
                                         'cluster_sizes for each VC'))
    cluster_size_args.add_argument('--cluster-size-output-file', type=str,
                                   default='cluster_sizes.json',
                                   help=('Output file for discovered '
                                         'cluster sizes'))
    cluster_size_args.add_argument('--util-threshold', type=float,
                                   default=0.95,
                                   help='Utilization threshold')
    args = parser.parse_args()
    main(args)
