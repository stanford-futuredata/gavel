import argparse
import datetime
import json
import io
import contextlib
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import sys

import job
from job_id_pair import JobIdPair
import policies
import scheduler
import utils

def emulate_with_timeout(experiment_id, policy_name, schedule_in_rounds,
                         throughputs_file, cluster_spec, lam, seed, interval,
                         jobs_to_complete, fixed_job_duration, log_dir, timeout,
                         completion_algo, measurement_percentage, verbose):
    f = io.StringIO()
    measurement_percentage_str =\
        'measurement_percentage=%2f.log' % (measurement_percentage)
    with open(os.path.join(log_dir, measurement_percentage_str), 'w') as f:
        with contextlib.redirect_stdout(f):
            policy = utils.get_policy(policy_name, seed)
            sched = scheduler.Scheduler(
                            policy,
                            schedule_in_rounds=schedule_in_rounds,
                            throughputs_file=throughputs_file,
                            seed=seed,
                            time_per_iteration=interval,
                            emulate=True,
                            predict_throughputs=True)
            sched.set_throughput_prediction_config(measurement_percentage,
                                                   completion_algo)

            cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                           cluster_spec['p100'],
                                                           cluster_spec['k80'])
            if verbose:
                current_time = datetime.datetime.now()
                print('[%s] [Experiment ID: %2d] '
                      'Configuration: cluster_spec=%s, policy=%s, '
                       'seed=%d, lam=%f, completion_algo=%s, '
                       'measurement_percentage=%.2f' % (current_time,
                                                        experiment_id,
                                                        cluster_spec_str,
                                                        policy.name,
                                                        seed,
                                                        lam,
                                                        completion_algo,
                                                        measurement_percentage),
                      file=sys.stderr)

            if timeout is None:
                sched.emulate(cluster_spec, lam=lam,
                              jobs_to_complete=jobs_to_complete,
                              fixed_job_duration=fixed_job_duration)
                average_jct = sched.get_average_jct(jobs_to_complete)
                utilization = sched.get_cluster_utilization()
            else:
                try:
                    func_timeout(timeout, sched.emulate,
                                 args=(cluster_spec,),
                                 kwargs={
                                    'lam': lam,
                                    'jobs_to_complete': jobs_to_complete,
                                    'fixed_job_duration': fixed_job_duration,
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
                                                           utilization),
              file=sys.stderr)

    return average_jct, utilization

def emulate_with_timeout_helper(args):
    emulate_with_timeout(*args)

def main(args):
    if args.window_start >= args.window_end:
        raise ValueError('Window start must be < than window end.')

    schedule_in_rounds = True
    throughputs_file = 'combined_throughputs.json'
    num_v100s = args.gpus
    policy_names = args.policies
    job_range = (args.window_start, args.window_end)
    experiment_id = 0
    measurement_percentages =\
        np.linspace(0.0, 1.0, num=args.num_measurement_percentages,
                    endpoint=True)[1:]

    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)

    raw_logs_dir = os.path.join(args.log_dir, 'raw_logs')
    if not os.path.isdir(raw_logs_dir):
        os.mkdir(raw_logs_dir)

    jobs_to_complete = set()
    for i in range(job_range[0], job_range[1]):
        jobs_to_complete.add(JobIdPair(i, None))

    all_args_list = []
    for ratio_str in args.ratios:
        ratio = {}
        x = ratio_str.split(':')
        if len(x) != 3:
            raise ValueError('Invalid cluster ratio %s' % (ratio_str))
        ratio = {
            'v100': int(x[0]),
            'p100': int(x[1]),
            'k80': int(x[2])
            }
        cluster_spec = {}
        total_gpu_fraction = sum([ratio[gpu_type] for gpu_type in ratio])
        for gpu_type in ratio:
            fraction = ratio[gpu_type] / total_gpu_fraction
            cluster_spec[gpu_type] = int(fraction * num_v100s)

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

            for completion_algo in args.completion_algorithms:
                completion_algo_str =\
                    'completion_algo=%s' % (completion_algo)
                raw_logs_completion_algo_subdir =\
                    os.path.join(raw_logs_policy_subdir,
                                 completion_algo_str)
                if not os.path.isdir(raw_logs_completion_algo_subdir):
                    os.mkdir(raw_logs_completion_algo_subdir)

                for seed in args.seeds:
                    seed_str = 'seed=%d' % (seed)
                    raw_logs_seed_subdir = \
                            os.path.join(raw_logs_completion_algo_subdir,
                                         seed_str)
                    if not os.path.isdir(raw_logs_seed_subdir):
                        os.mkdir(raw_logs_seed_subdir)

                    for measurement_percentage in measurement_percentages:
                        all_args_list.append((experiment_id, policy_name,
                                              schedule_in_rounds,
                                              throughputs_file, cluster_spec,
                                              args.lam, seed, args.interval,
                                              jobs_to_complete,
                                              args.fixed_job_duration,
                                              raw_logs_seed_subdir,
                                              args.timeout,
                                              completion_algo,
                                              measurement_percentage,
                                              args.verbose))
                        experiment_id += 1

    if len(all_args_list) > 0:
        current_time = datetime.datetime.now()
        print('[%s] Running %d total experiment(s)...' % (current_time,
                                                          len(all_args_list)))
        with multiprocessing.Pool(args.processes) as p:
            p.map(emulate_with_timeout_helper, all_args_list)
    else:
        raise ValueError('No work to be done!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Sweep through throughput estimation parameters.')

    parser.add_argument('-g', '--gpus', type=int, default=25,
                        help='Number of v100 GPUs')
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
                        choices=['fifo_packed', 'max_min_fairness_packed',
                                 'min_total_duration_packed'],
                        default=['fifo_packed', 'max_min_fairness_packed',
                                 'min_total_duration_packed'],
                        help='List of policies to sweep')
    parser.add_argument('-r', '--ratios', type=str, nargs='+',
                        default=['1:0:0', '1:1:0', '1:1:1', '2:1:0'],
                        help=('List of cluster ratios to sweep in the form '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--lam', type=float, default=4096,
                        help='Lambda value to fix for all experiments.')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 42, 1234, 10],
                        help='List of random seeds')
    parser.add_argument('-i', '--interval', type=int, default=1920,
                        help='Interval length (in seconds)')
    parser.add_argument('-f', '--fixed-job-duration', type=int, default=None,
                        help=('If set, fixes the duration of all jobs to the '
                              'specified value (in seconds)'))
    parser.add_argument('--completion_algorithms', type=str, nargs='+',
                        choices=['NN', 'SVT', 'PMF', 'BMF'],
                        default=['SVT', 'NN', 'PMF', 'BMF'],
                        help=('Matrix completion algorithm for throughput '
                              'prediction'))
    parser.add_argument('--num_measurement_percentages', type=int, default=20,
                        help=('Number job pair measurement percentages '
                              'to sweep.'))
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose')
    args = parser.parse_args()
    main(args)
