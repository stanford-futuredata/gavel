import argparse
import datetime
import json
import io
from contextlib import redirect_stdout
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import numpy as np
import os

import job
from job_id_pair import JobIdPair
import policies
import scheduler

UTILIZATION_THRESHOLD = .98

np.random.seed(42)

def get_policy(policy_name):
    if policy_name == "isolated":
        policy = policies.IsolatedPolicy()
    elif policy_name == "max_min_fairness":
        policy = policies.MaxMinFairnessPolicy()
    elif policy_name == "max_min_fairness_packed":
        policy = policies.MaxMinFairnessPolicyWithPacking()
    elif policy_name == "min_total_duration":
        policy = policies.MinTotalDurationPolicy()
    elif policy_name == "min_total_duration_packed":
        policy = policies.MinTotalDurationPolicyWithPacking()
    elif policy_name == "fifo":
        policy = policies.FIFOPolicy()
    else:
        raise Exception("Unknown policy!")
    return policy

def emulate_with_timeout(policy_name, schedule_in_rounds, throughputs_file,
                         cluster_spec, lam, jobs_to_complete, log_dir,
                         timeout=3600):
    policy = get_policy(policy_name)
    sched = scheduler.Scheduler(
                    policy,
                    schedule_in_rounds=schedule_in_rounds,
                    throughputs_file=throughputs_file,
                    emulate=True)

    cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                   cluster_spec['p100'],
                                                   cluster_spec['k80'])
    print('%s] cluster_spec=%s, policy=%s, lam=%f' % (datetime.datetime.now(),
                                                      cluster_spec_str,
                                                      policy.name, lam))

    f = io.StringIO()
    with redirect_stdout(f):
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

    print('%s] average JCT=%f, utilization=%f' % (datetime.datetime.now(),
                                                  average_jct, utilization))

    return average_jct, utilization

def main(args):
    if args.window_start >= args.window_end:
        raise ValueError('Window start must be < than window end')
    schedule_in_rounds = True
    throughputs_file = 'combined_throughputs.json'
    num_v100s = 25
    policy_names = ['max_min_fairness', 'fifo', 'isolated']
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

            all_lams = []
            average_jcts = []
            utilizations = []
            schedule_in_rounds = policy_name != 'fifo'

            # Sweep all power of 2 lambdas until utilization == 1.0.
            lam = 32768
            while True:
                all_lams.append(lam)
                average_jct, utilization = \
                        emulate_with_timeout(policy_name, schedule_in_rounds,
                                             throughputs_file, cluster_spec,
                                             lam, jobs_to_complete,
                                             raw_logs_policy_subdir)

                average_jcts.append(average_jct)
                utilizations.append(utilization)
                if utilization < UTILIZATION_THRESHOLD:
                    lam /= 2
                else:
                    break

            # Find the knee of the throughput vs latency plot.
            lams = np.linspace(lam * 2, lam, num=10)[1:]
            for lam in lams:
                all_lams.append(lam)
                average_jct, utilization = \
                        emulate_with_timeout(policy_name, schedule_in_rounds,
                                             throughputs_file, cluster_spec,
                                             lam, jobs_to_complete,
                                             raw_logs_policy_subdir)

                average_jcts.append(average_jct)
                utilizations.append(utilization)
                if utilization >= UTILIZATION_THRESHOLD:
                    knee = lam
                    break

            # Extend the throughput vs latency plot until the latency under
            # high load is an order of magnitude larger than the latency under
            # low load.
            i = 1
            while True:
                lam = knee * (1.0 - i * .05)
                all_lams.append(lam)
                average_jct, utilization = \
                        emulate_with_timeout(policy_name, schedule_in_rounds,
                                             throughputs_file, cluster_spec,
                                             lam, jobs_to_complete,
                                             raw_logs_policy_subdir)
                average_jcts.append(average_jct)
                utilizations.append(utilization)
                if np.max(average_jcts) / np.min(average_jcts) >= 10:
                    break

            print('knee at lamda=', knee)
            print('final lambda=', lam)
            print(all_lams)
            print(average_jcts)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Sweep through lambda values')
    parser.add_argument('-v', '--v100s', type=int, default=25,
                        help='Number of v100s')
    parser.add_argument('-n', '--num_jobs', type=int, default=1000,
                        help='Number of jobs to run')
    parser.add_argument('-l', '--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('-s', '--window_start', type=int, default=1000,
                        help='Measurement window start (job ID)')
    parser.add_argument('-e', '--window_end', type=int, default=2000,
                        help='Measurement window end (job ID)')

    args = parser.parse_args()
    main(args)
