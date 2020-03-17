import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import json
import contextlib
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing
import numpy as np
import os
import random
import sys
import time

from job_id_pair import JobIdPair
import scheduler
import utils


def simulate_with_timeout(experiment_id, policy_name,
                          throughputs_file, per_instance_type_prices_dir,
                          available_clouds, assign_SLAs, cluster_spec, lam,
                          seed, interval, fixed_job_duration,
                          generate_multi_gpu_jobs, num_total_jobs, solver,
                          log_dir, timeout, verbose):
    # Add some random delay to prevent outputs from overlapping.
    # TODO: Replace this with postprocessing in the log parsing script.
    time.sleep(random.uniform(0, 5))
    num_total_jobs_str = 'num_total_jobs=%d.log' % (num_total_jobs)
    with open(os.path.join(log_dir, num_total_jobs_str), 'w') as f:
        with contextlib.redirect_stdout(f):
            policy = utils.get_policy(policy_name, seed=seed, solver=solver)
            sched = \
                scheduler.Scheduler(
                    policy, throughputs_file=throughputs_file,
                    seed=seed, time_per_iteration=interval,
                    per_instance_type_prices_dir=per_instance_type_prices_dir,
                    available_clouds = available_clouds,
                    assign_SLAs=assign_SLAs,
                    simulate=True)

            cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                           cluster_spec['p100'],
                                                           cluster_spec['k80'])
            if verbose:
                current_time = datetime.datetime.now()
                print('[%s] [Experiment ID: %2d] '
                      'Configuration: cluster_spec=%s, policy=%s, '
                       'seed=%d, num_total_jobs=%d' % (current_time,
                                                       experiment_id,
                                                       cluster_spec_str,
                                                       policy.name,
                                                       seed, num_total_jobs),
                      file=sys.stderr)

            if timeout is None:
                sched.simulate(cluster_spec, lam=lam,
                               fixed_job_duration=fixed_job_duration,
                               generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                               num_total_jobs=num_total_jobs)
                average_jct = sched.get_average_jct()
                utilization = sched.get_cluster_utilization()
                makespan = sched.get_current_timestamp()
                total_cost = sched.get_total_cost()
            else:
                try:
                    func_timeout(timeout, sched.simulate,
                                 args=(cluster_spec,),
                                 kwargs={
                                    'lam': lam,
                                    'fixed_job_duration': fixed_job_duration,
                                    'generate_multi_gpu_jobs': generate_multi_gpu_jobs,
                                    'num_total_jobs': num_total_jobs
                                 })
                    average_jct = sched.get_average_jct()
                    utilization = sched.get_cluster_utilization()
                    makespan = sched.get_current_timestamp()
                    total_cost = sched.get_total_cost()
                except FunctionTimedOut:
                    average_jct = float('inf')
                    utilization = 1.0
                    makespan = float('inf')
                    total_cost = float('inf')

    if verbose:
        current_time = datetime.datetime.now()
        print('[%s] [Experiment ID: %2d] '
              'Results: average JCT=%f, utilization=%f, '
              'makespan=%f, total_cost=$%.2f' % (
                  current_time,
                  experiment_id,
                  average_jct,
                  utilization,
                  makespan,
                  total_cost),
              file=sys.stderr)

    return average_jct, utilization

def main(args):
    if ((args.num_total_jobs_lower_bound is None and
         args.num_total_jobs_upper_bound is not None) or
        (args.num_total_jobs_lower_bound is not None and
         args.num_total_jobs_upper_bound is None)):
        raise ValueError('If num_total_jobs range is not None, both '
                         'bounds must be specified.')
    throughputs_file = args.throughputs_file
    policy_names = args.policies
    experiment_id = 0

    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)

    raw_logs_dir = os.path.join(args.log_dir, 'raw_logs')
    if not os.path.isdir(raw_logs_dir):
        os.mkdir(raw_logs_dir)

    all_args_list = []
    for cluster_spec_str in args.cluster_spec:
        cluster_spec_str_split = cluster_spec_str.split(':')
        if len(cluster_spec_str_split) != 3:
            raise ValueError('Invalid cluster spec %s' % (cluster_spec_str))
        cluster_spec = {
            'v100': int(cluster_spec_str_split[0]),
            'p100': int(cluster_spec_str_split[1]),
            'k80': int(cluster_spec_str_split[2]),
        }

        cluster_spec_str = 'v100=%d.p100=%d.k80=%d' % (cluster_spec['v100'],
                                                       cluster_spec['p100'],
                                                       cluster_spec['k80'])
        raw_logs_cluster_spec_subdir = os.path.join(raw_logs_dir,
                                                    cluster_spec_str)
        if not os.path.isdir(raw_logs_cluster_spec_subdir):
            os.mkdir(raw_logs_cluster_spec_subdir)

        for policy_name in policy_names:
            raw_logs_policy_subdir = os.path.join(raw_logs_cluster_spec_subdir,
                                                  policy_name)
            if not os.path.isdir(raw_logs_policy_subdir):
                os.mkdir(raw_logs_policy_subdir)

            lower_bound = args.num_total_jobs_lower_bound
            upper_bound = args.num_total_jobs_upper_bound
            all_num_total_jobs = np.linspace(lower_bound, upper_bound,
                                             args.num_data_points)
            if all_num_total_jobs[0] == 0:
                all_num_total_jobs = all_num_total_jobs[1:]
            for num_total_jobs in all_num_total_jobs:
                lam = 0.0  # All jobs are added at the start of the trace.
                for seed in args.seeds:
                    seed_str = 'seed=%d' % (seed)
                    raw_logs_seed_subdir = \
                            os.path.join(raw_logs_policy_subdir, seed_str)
                    if not os.path.isdir(raw_logs_seed_subdir):
                        os.mkdir(raw_logs_seed_subdir)
                    all_args_list.append((experiment_id, policy_name,
                                          throughputs_file,
                                          args.per_instance_type_prices_dir,
                                          args.available_clouds,
                                          args.assign_SLAs,
                                          cluster_spec,
                                          lam, seed, args.interval,
                                          args.fixed_job_duration,
                                          args.generate_multi_gpu_jobs,
                                          num_total_jobs,
                                          args.solver,
                                          raw_logs_seed_subdir,
                                          args.timeout, args.verbose))
                    experiment_id += 1
    if len(all_args_list) > 0:
        current_time = datetime.datetime.now()
        print('[%s] Running %d total experiment(s)...' % (current_time,
                                                          len(all_args_list)))
        with multiprocessing.Pool(args.processes) as p:
            # Sort args in order of increasing num_total_jobs to prioritize
            # short-running jobs.
            all_args_list.sort(key=lambda x: x[12])
            results = [p.apply_async(simulate_with_timeout, args_list)
                       for args_list in all_args_list]
            results = [result.get() for result in results]
    else:
        raise ValueError('No work to be done!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Sweep through lambda values')
    fixed_range = parser.add_argument_group('Sweep over fixed range')

    parser.add_argument('-l', '--log-dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('-t', '--timeout', type=int, default=None,
                        help='Timeout (in seconds) for each run')
    parser.add_argument('-j', '--processes', type=int, default=None,
                        help=('Number of processes to use in pool '
                              '(use as many as available if not specified)'))
    parser.add_argument('-p', '--policies', type=str, nargs='+',
                        default=utils.get_available_policies(),
                        help='List of policies to sweep')
    parser.add_argument('-c', '--cluster-spec', type=str, nargs='+',
                        default=['25:0:0', '12:12:0', '16:8:0', '8:8:8'],
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4],
                        help='List of random seeds')
    parser.add_argument('-i', '--interval', type=int, default=1920,
                        help='Interval length (in seconds)')
    parser.add_argument('-f', '--fixed-job-duration', type=int, default=None,
                        help=('If set, fixes the duration of all jobs to the '
                              'specified value (in seconds)'))
    parser.add_argument('--throughputs-file', type=str,
                        default=('/lfs/1/keshav2/gpusched/scheduler/'
                                 'oracle_throughputs.json'),
                        help='Oracle throughputs file')
    parser.add_argument('-m', '--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose')
    parser.add_argument('--per_instance_type_prices_dir', type=str,
                        default=None,
                        help='Per-instance-type prices directory')
    parser.add_argument('--available_clouds', type=str, nargs='+',
                        choices=['aws', 'gcp', 'azure'],
                        default=['aws', 'gcp', 'azure'],
                        help='Clouds available to rent machines from')
    parser.add_argument('--assign_SLAs', action='store_true', default=False,
                        help='If set, assigns SLAs to each job')
    fixed_range.add_argument('-a', '--num-total-jobs-lower-bound', type=int,
                             default=None,
                             help='Lower bound for num_total_jobs to sweep')
    fixed_range.add_argument('-b', '--num-total-jobs-upper-bound', type=int,
                             default=None,
                             help='Upper bound for num_total_jobs to sweep')
    fixed_range.add_argument('-n', '--num-data-points', type=int, default=20,
                             help='Number of data points to sweep through')
    args = parser.parse_args()
    main(args)
