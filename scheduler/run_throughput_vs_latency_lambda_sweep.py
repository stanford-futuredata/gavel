import argparse
import datetime
import json
import io
from contextlib import redirect_stdout
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os

import job
from job_id_pair import JobIdPair
import policies
import scheduler

def get_policy(policy_name):
    if policy_name == 'isolated':
        policy = policies.IsolatedPolicy()
    elif policy_name == 'max_min_fairness':
        policy = policies.MaxMinFairnessPolicy()
    elif policy_name == 'max_min_fairness_packed':
        policy = policies.MaxMinFairnessPolicyWithPacking()
    elif policy_name == 'min_total_duration':
        policy = policies.MinTotalDurationPolicy()
    elif policy_name == 'min_total_duration_packed':
        policy = policies.MinTotalDurationPolicyWithPacking()
    elif policy_name == 'fifo':
        policy = policies.FIFOPolicy()
    else:
        raise Exception('Unknown policy!')
    return policy

def emulate_with_timeout(policy_name, schedule_in_rounds, throughputs_file,
                         cluster_spec, lam, jobs_to_complete, log_dir, timeout,
                         verbose):
    policy = get_policy(policy_name)
    sched = scheduler.Scheduler(
                    policy,
                    schedule_in_rounds=schedule_in_rounds,
                    throughputs_file=throughputs_file,
                    emulate=True)

    cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                   cluster_spec['p100'],
                                                   cluster_spec['k80'])
    if verbose:
        current_time = datetime.datetime.now()
        print('%s] cluster_spec=%s, policy=%s, lam=%f' % (current_time,
                                                          cluster_spec_str,
                                                          policy.name, lam))

    f = io.StringIO()
    with redirect_stdout(f):
        if timeout is None:
            sched.emulate(cluster_spec, lam=lam,
                          jobs_to_complete=jobs_to_complete)
            average_jct = sched.get_average_jct(jobs_to_complete)
            utilization = sched.get_cluster_utilization()
        else:
            try:
                func_timeout(timeout, sched.emulate,
                             args=(cluster_spec,),
                             kwargs={
                                'lam': lam,
                                'jobs_to_complete': jobs_to_complete
                             })
                average_jct = sched.get_average_jct(jobs_to_complete)
                utilization = sched.get_cluster_utilization()
            except FunctionTimedOut:
                average_jct = float('inf')
                utilization = 1.0
    output = f.getvalue()
    lam_str = 'lambda=%f.log' % (lam)
    with open(os.path.join(log_dir, lam_str), 'w') as f:
        f.write(output)

    if verbose:
        current_time = datetime.datetime.now()
        print('%s] average JCT=%f, utilization=%f' % (current_time, average_jct,
                                                      utilization))

    return average_jct, utilization

def emulate_with_timeout_helper(args):
    emulate_with_timeout(*args)

def run_automatic_sweep(policy_name, schedule_in_rounds, throughputs_file,
                        cluster_spec, jobs_to_complete, raw_logs_policy_subdir,
                        timeout, verbose):
    all_lams = []
    average_jcts = []
    utilizations = []

    # Sweep all power of 2 lambdas until utilization == 1.0.
    lam = 32768
    while True:
        all_lams.append(lam)
        average_jct, utilization = \
                emulate_with_timeout(policy_name,
                                     schedule_in_rounds,
                                     throughputs_file, cluster_spec,
                                     lam, jobs_to_complete,
                                     raw_logs_policy_subdir,
                                     timeout, verbose)

        average_jcts.append(average_jct)
        utilizations.append(utilization)
        if utilization < args.utilization_threshold:
            lam /= 2
        else:
            break

    # Find the knee of the throughput vs latency plot.
    lams = np.linspace(lam * 2, lam, num=10)[1:]
    for lam in lams:
        all_lams.append(lam)
        average_jct, utilization = \
                emulate_with_timeout(policy_name,
                                     schedule_in_rounds,
                                     throughputs_file, cluster_spec,
                                     lam, jobs_to_complete,
                                     raw_logs_policy_subdir,
                                     timeout, verbose)

        average_jcts.append(average_jct)
        utilizations.append(utilization)
        if utilization >= args.utilization_threshold:
            knee = lam
            break

    # Extend the throughput vs latency plot until the latency under
    # high load is an order of magnitude larger than the latency
    # under low load.
    i = 1
    while True:
        lam = knee * (1.0 - i * .05)
        all_lams.append(lam)
        average_jct, utilization = \
                emulate_with_timeout(policy_name,
                                     schedule_in_rounds,
                                     throughputs_file, cluster_spec,
                                     lam, jobs_to_complete,
                                     raw_logs_policy_subdir,
                                     timeout, verbose)
        average_jcts.append(average_jct)
        utilizations.append(utilization)
        if np.max(average_jcts) / np.min(average_jcts) >= 10:
            break

    print('knee at lamda=', knee)
    print('final lambda=', lam)
    for lam, average_jct, utilization in \
            zip(all_lams, average_jcts, utilizations):
        print('Lambda=%f,Average JCT=%f,'
              'Utilization=%f' % (lam, average_jct, utilization))

def run_automatic_sweep_helper(args):
    run_automatic_sweep(*args)

def main(args):
    if args.window_start >= args.window_end:
        raise ValueError('Window start must be < than window end.')
    if ((args.throughput_lower_bound is None and
         args.throughput_upper_bound is not None) or
        (args.throughput_lower_bound is not None and
         args.throughput_upper_bound is None)):
        raise ValueError('If throughput range is not None, both '
                         'bounds must be specified.')
    elif (args.throughput_lower_bound is not None and
          args.throughput_upper_bound is not None):
        automatic_sweep = False
    else:
        automatic_sweep = True
    schedule_in_rounds = True
    throughputs_file = 'combined_throughputs.json'
    num_v100s = args.gpus
    policy_names = ['fifo']
    ratios = [
            {'v100': 1, 'p100': 0, 'k80': 0},
            {'v100': 1, 'p100': 1, 'k80': 0},
            {'v100': 1, 'p100': 1, 'k80': 1},
            {'v100': 2, 'p100': 1, 'k80': 0},
        ]
    job_range = (args.window_start, args.window_end)

    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)

    raw_logs_dir = os.path.join(args.log_dir, 'raw_logs')
    if not os.path.isdir(raw_logs_dir):
        os.mkdir(raw_logs_dir)

    jobs_to_complete = set()
    for i in range(job_range[0], job_range[1]):
        jobs_to_complete.add(JobIdPair(i, None))

    all_args_list = []
    for ratio in ratios:
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

            if automatic_sweep:
                all_args_list.append((policy_name, schedule_in_rounds,
                                      throughputs_file, cluster_spec,
                                      jobs_to_complete, raw_logs_policy_subdir,
                                      args.timeout, args.verbose))
            else:
                throughputs = list(np.linspace(args.throughput_lower_bound,
                                               args.throughput_upper_bound,
                                               num=args.num_data_points))
                if throughputs[0] == 0.0:
                    throughputs = throughputs[1:]
                for throughput in throughputs:
                    lam = 3600.0 / throughput
                    all_args_list.append((policy_name, schedule_in_rounds,
                                          throughputs_file, cluster_spec,
                                          lam, jobs_to_complete,
                                          raw_logs_policy_subdir,
                                          args.timeout, args.verbose))
    if len(all_args_list) > 0:
        with multiprocessing.Pool(args.processes) as p:
            if automatic_sweep:
                p.map(run_automatic_sweep_helper, all_args_list)
            else:
                p.map(emulate_with_timeout_helper, all_args_list)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Sweep through lambda values')
    automatic = parser.add_argument_group('Automatic sweep')
    fixed_range = parser.add_argument_group('Sweep over fixed range')

    parser.add_argument('-g', '--gpus', type=int, default=25,
                        help='Number of v100 GPUs')
    parser.add_argument('-l', '--log-dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('-s', '--window-start', type=int, default=0,
                        help='Measurement window start (job ID)')
    parser.add_argument('-e', '--window-end', type=int, default=5000,
                        help='Measurement window end (job ID)')
    parser.add_argument('-t', '--timeout', type=int, default=3600,
                        help='Timeout (in seconds) for each run')
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help=('Number of processes to use in pool '
                              '(use as many as available if not specified)'))
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Verbose')
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
    automatic.add_argument('-u', '--utilization-threshold', type=float,
                           default=.98,
                           help=('Utilization threshold to use when '
                                 'automatically sweeping lambdas'))
    args = parser.parse_args()
    main(args)
