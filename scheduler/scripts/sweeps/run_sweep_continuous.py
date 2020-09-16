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
import sys

from job_id_pair import JobIdPair
from job_table import JobTable
import scheduler
import utils


def simulate_with_timeout(experiment_id, policy_name,
                          throughputs_file, cluster_spec, lam, seed, interval,
                          jobs_to_complete, fixed_job_duration, solver,
                          generate_multi_gpu_jobs,
                          generate_multi_priority_jobs, simulate_steady_state,
                          log_dir, timeout, verbose, checkpoint_threshold,
                          profiling_percentage, num_reference_models,
                          num_gpus_per_server, ideal):
    lam_str = 'lambda=%f.log' % (lam)
    checkpoint_file = None
    if checkpoint_threshold is not None:
        checkpoint_file = os.path.join(log_dir, 'lambda=%f.pickle' % lam)

    cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                           cluster_spec['p100'],
                                                           cluster_spec['k80'])
    policy = utils.get_policy(policy_name, solver=solver, seed=seed)
    if verbose:
        current_time = datetime.datetime.now()
        print('[%s] [Experiment ID: %2d] '
              'Configuration: cluster_spec=%s, policy=%s, '
              'seed=%d, lam=%f, '
              'profiling_percentage=%f, '
              'num_reference_models=%d' % (current_time,
                                           experiment_id,
                                           cluster_spec_str,
                                           policy.name,
                                           seed, lam,
                                           profiling_percentage,
                                           num_reference_models))

    with open(os.path.join(log_dir, lam_str), 'w') as f:
        with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
            sched = scheduler.Scheduler(
                            policy,
                            throughputs_file=throughputs_file,
                            seed=seed,
                            time_per_iteration=interval,
                            simulate=True,
                            profiling_percentage=profiling_percentage,
                            num_reference_models=num_reference_models)

            if timeout is None:
                sched.simulate(cluster_spec, lam=lam,
                               jobs_to_complete=jobs_to_complete,
                               fixed_job_duration=fixed_job_duration,
                               generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                               generate_multi_priority_jobs=generate_multi_priority_jobs,
                               simulate_steady_state=simulate_steady_state,
                               checkpoint_file=checkpoint_file,
                               checkpoint_threshold=checkpoint_threshold,
                               num_gpus_per_server=num_gpus_per_server,
                               ideal=ideal)
                average_jct = sched.get_average_jct(jobs_to_complete)
                utilization = 1.0
                if not ideal:
                    utilization = sched.get_cluster_utilization()
            else:
                try:
                    func_timeout(timeout, sched.simulate,
                                 args=(cluster_spec,),
                                 kwargs={
                                    'lam': lam,
                                    'jobs_to_complete': jobs_to_complete,
                                    'fixed_job_duration': fixed_job_duration,
                                    'generate_multi_gpu_jobs': generate_multi_gpu_jobs,
                                    'generate_multi_priority_jobs': generate_multi_priority_jobs,
                                    'simulate_steady_state': simulate_steady_state,
                                    'checkpoint_file': checkpoint_file,
                                    'checkpoint_threshold': checkpoint_threshold,
                                    'num_gpus_per_server': num_gpus_per_server,
                                    'ideal': ideal
                                 })
                    average_jct = sched.get_average_jct(jobs_to_complete)
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
                                                           utilization))
    sched.shutdown()

    return average_jct, utilization

def main(args):
    if args.window_start >= args.window_end:
        raise ValueError('Window start must be < than window end.')
    if (args.throughput_lower_bound is None or
        args.throughput_upper_bound is None):
        raise ValueError('Throughput range must be specified.')

    cutoff_throughputs = {}
    if args.cutoff_throughputs_file is not None:
        cutoff_throughputs = json.load(open(args.cutoff_throughputs_file, 'r'))

    throughputs_file = args.throughputs_file
    policy_names = args.policies
    profiling_percentages = args.profiling_percentages
    all_num_reference_models = args.num_reference_models
    estimate_throughputs = (min(profiling_percentages) < 1.0 or
                            min(all_num_reference_models) < len(JobTable))
    job_range = (args.window_start, args.window_end)
    experiment_id = 0

    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)

    raw_logs_dir = os.path.join(args.log_dir, 'raw_logs')
    if not os.path.isdir(raw_logs_dir):
        os.mkdir(raw_logs_dir)

    jobs_to_complete = set()
    for i in range(job_range[0], job_range[1]):
        jobs_to_complete.add(JobIdPair(i, None))

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
        num_gpus_per_server_split = args.num_gpus_per_server.split(':')
        num_gpus_per_server = {
            'v100': int(num_gpus_per_server_split[0]),
            'p100': int(num_gpus_per_server_split[1]),
            'k80': int(num_gpus_per_server_split[2]),
        }

        raw_logs_cluster_spec_subdir = \
            os.path.join(raw_logs_dir,
                         'v100=%d.p100=%d.k80=%d' % (cluster_spec['v100'],
                                                     cluster_spec['p100'],
                                                     cluster_spec['k80']))
        if not os.path.isdir(raw_logs_cluster_spec_subdir):
            os.mkdir(raw_logs_cluster_spec_subdir)

        for policy_name in policy_names:
            raw_logs_policy_subdir = os.path.join(raw_logs_cluster_spec_subdir,
                                           policy_name)
            if not os.path.isdir(raw_logs_policy_subdir):
                os.mkdir(raw_logs_policy_subdir)

            for profiling_percentage in profiling_percentages:
                if estimate_throughputs:
                    profiling_percentage_str = \
                        'profiling_percentage=%f' % (profiling_percentage)
                    raw_logs_profiling_subdir = \
                        os.path.join(raw_logs_policy_subdir,
                                     profiling_percentage_str)
                    if not os.path.isdir(raw_logs_profiling_subdir):
                        os.mkdir(raw_logs_profiling_subdir)
                else:
                    raw_logs_profiling_subdir = raw_logs_policy_subdir
                for i, num_reference_models in enumerate(args.num_reference_models):
                    if estimate_throughputs:
                        num_reference_models_str = \
                            'num_reference_models=%d' % (num_reference_models)
                        raw_logs_num_reference_models_subdir = \
                            os.path.join(raw_logs_profiling_subdir,
                                         num_reference_models_str)
                        if not os.path.isdir(raw_logs_num_reference_models_subdir):
                            os.mkdir(raw_logs_num_reference_models_subdir)
                    else:
                        raw_logs_num_reference_models_subdir = \
                            raw_logs_policy_subdir
                    throughputs = \
                        list(np.linspace(args.throughput_lower_bound,
                                         args.throughput_upper_bound,
                                         num=args.num_data_points))
                    if throughputs[0] == 0.0:
                        throughputs = throughputs[1:]
                    for throughput in throughputs:
                        if (cluster_spec_str in cutoff_throughputs and
                            policy_name in cutoff_throughputs[cluster_spec_str]):
                            cutoff_throughput = \
                                cutoff_throughputs[cluster_spec_str][policy_name]
                            if throughput >= cutoff_throughput:
                                print('Throughput of %f is too high '
                                      'for policy %s with cluster '
                                      'spec %s.' % (throughput,
                                                    policy_name,
                                                    cluster_spec_str))
                                continue

                        lam = 3600.0 / throughput
                        for seed in args.seeds:
                            seed_str = 'seed=%d' % (seed)
                            raw_logs_seed_subdir = os.path.join(
                                    raw_logs_num_reference_models_subdir,
                                    seed_str)
                            if not os.path.isdir(raw_logs_seed_subdir):
                                os.mkdir(raw_logs_seed_subdir)
                            all_args_list.append((experiment_id, policy_name,
                                                  throughputs_file,
                                                  cluster_spec,
                                                  lam, seed, args.interval,
                                                  jobs_to_complete,
                                                  args.fixed_job_duration,
                                                  args.solver,
                                                  args.generate_multi_gpu_jobs,
                                                  args.generate_multi_priority_jobs,
                                                  args.simulate_steady_state,
                                                  raw_logs_seed_subdir,
                                                  args.timeout,
                                                  args.verbose,
                                                  args.checkpoint_threshold,
                                                  profiling_percentage,
                                                  num_reference_models,
                                                  num_gpus_per_server,
                                                  args.ideal))
                            experiment_id += 1
    if len(all_args_list) > 0:
        current_time = datetime.datetime.now()
        print('[%s] Running %d total experiment(s)...' % (current_time,
                                                          len(all_args_list)))
        with multiprocessing.Pool(args.processes) as p:
            # Sort args in order of decreasing lambda to prioritize
            # short-running jobs.
            all_args_list.sort(key=lambda x: x[4], reverse=True)
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
    parser.add_argument('-s', '--window-start', type=int, default=0,
                        help='Measurement window start (job ID)')
    parser.add_argument('-e', '--window-end', type=int, default=5000,
                        help='Measurement window end (job ID)')
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
    parser.add_argument('--num_gpus_per_server', type=str, default='1:1:1',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4],
                        help='List of random seeds')
    parser.add_argument('-i', '--interval', type=int, default=360,
                        help='Interval length (in seconds)')
    parser.add_argument('-f', '--fixed-job-duration', type=int, default=None,
                        help=('If set, fixes the duration of all jobs to the '
                              'specified value (in seconds)'))
    parser.add_argument('--cutoff-throughputs-file', type=str, default=None,
                        help=('If set, uses the attached cutoff_throughputs '
                              'JSON file in sweep to limit args run'))
    parser.add_argument('--throughputs-file', type=str,
                        default='simulation_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('-m', '--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('--simulate-steady-state', action='store_true',
                        default=False,
                        help=('If set, adds as many jobs as there are workers '
                              'before beginning the simulation.'))
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose')
    parser.add_argument('--checkpoint-threshold', type=int, default=None,
                        help=('Checkpoint threshold, None if checkpointing is '
                              'disabled. Checkpoint is created after this '
                              'job ID is added.'))
    parser.add_argument('--profiling_percentages', type=float, nargs='+',
                        default=[1.0],
                        help=('Percentages of machines dedicated to profiling '
                              'co-located job pairs'))
    parser.add_argument('--num_reference_models', type=int, nargs='+',
                        default=[len(JobTable)],
                        help=('Number of reference models to use when '
                              'estimating throughputs'))
    parser.add_argument('--ideal', action='store_true', default=False,
                        help='Run allocations 100%% ideally')
    fixed_range.add_argument('-a', '--throughput-lower-bound', type=float,
                             default=None,
                             help=('Lower bound for throughput interval to '
                                   'sweep'))
    fixed_range.add_argument('-b', '--throughput-upper-bound', type=float,
                             default=None,
                             help=('Upper bound for throughput interval to '
                                   'sweep'))
    fixed_range.add_argument('-n', '--num-data-points', type=int, default=20,
                             help='Number of data points to sweep through')
    args = parser.parse_args()
    main(args)
