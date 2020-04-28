import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import numpy as np
import random
import time

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable


def _generate_job(rng, oracle_throughputs, generate_multi_gpu_jobs=False,
                  generate_multi_priority_jobs=False,
                  run_dir='/tmp'):
    """Generates a new job for simulation."""
    job_template = rng.choice(JobTable)
    job_type = job_template.model
    run_time = 60 * (10 ** random.uniform(2, 4))
    num_steps = \
        run_time * oracle_throughputs['v100'][job_type]['null']
    assert(run_time > 0)
    assert(num_steps > 0)
    if job_template.needs_data_dir:
        command = job_template.command % (run_dir, run_dir)
    else:
        command = job_template.command % (run_dir)

    scale_factor = 1
    if generate_multi_gpu_jobs:  # Copies Philly distribution.
        r = random.uniform(0, 1)
        if 0.8 <= r <= 0.85:
            scale_factor = 2
        elif 0.85 <= r <= 0.95:
            scale_factor = 4
        elif 0.95 <= r:
            scale_factor = 8

    priority_weight = 1.0
    if generate_multi_priority_jobs:
        r = random.uniform(0, 1)
        if 0.0 <= r <= 0.2:
            priority_weight = 5.0

    job = Job(job_id=None,
              job_type=job_type,
              command=command,
              num_steps_arg=job_template.num_steps_arg,
              total_steps=num_steps,
              duration=None,
              scale_factor=scale_factor,
              priority_weight=priority_weight)

    return job

def measure_runtime(cluster_spec_str, num_active_jobs, policy_name,
                    oracle_throughputs, generate_multi_gpu_jobs,
                    generate_multi_priority_jobs, num_trials):
    policy = utils.get_policy(policy_name)
    cluster_spec = {}
    v100s, p100s, k80s = cluster_spec_str.split(':')
    cluster_spec = {
        'v100': int(v100s),
        'p100': int(p100s),
        'k80': int(k80s),
    }

    # TODO: support sweeping seeds?
    rng = random.Random()
    rng.seed(0)
    throughputs = {}
    jobs = {}
    for i in range(num_active_jobs):
        job_id = JobIdPair(i, None)
        jobs[i] = _generate_job(rng, oracle_throughputs,
                                generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                                generate_multi_priority_jobs=generate_multi_priority_jobs)
        throughputs[job_id] = {}
        for worker_type in cluster_spec:
            throughputs[job_id][worker_type] = \
                oracle_throughputs[worker_type][jobs[i].job_type]['null']
    if '_Packing' in policy.name:
        for i in range(num_active_jobs):
            for j in range(num_active_jobs):
                if i < j and jobs[i].scale_factor == jobs[j].scale_factor:
                    throughputs[JobIdPair(i, j)] = {}
                    for worker_type in cluster_spec:
                        throughputs[JobIdPair(i, j)][worker_type] = \
                            oracle_throughputs[worker_type][jobs[i].job_type][jobs[j].job_type]
    scale_factors = {
        JobIdPair(i, None): jobs[i].scale_factor for i in range(num_active_jobs)
    }
    results = []
    for trial in range(num_trials):
        start_time = time.time()
        if policy.name.startswith('MaxMinFairness'):
            priority_weights = {
                JobIdPair(i, None): jobs[i].priority_weight for i in range(num_active_jobs)
            }
            policy.get_allocation(
                throughputs, scale_factors, priority_weights,
                cluster_spec)
        elif policy.name.startswith('MinTotalDuration'):
            num_steps_remaining = {
                JobIdPair(i, None): jobs[i].num_steps for i in range(num_active_jobs)
            }
            policy.get_allocation(
                throughputs, scale_factors, num_steps_remaining,
                cluster_spec)
        else:
            policy.get_allocation(
                throughputs, scale_factors, cluster_spec)

        runtime = time.time() - start_time
        results.append(runtime)
    print('%s,%d,%s,%d,%f,%f' % (cluster_spec_str, num_active_jobs,
                                 policy.name, num_trials, np.mean(results),
                                 np.std(results)))

def main(args):
    all_cluster_specs = args.cluster_spec
    all_num_active_jobs = args.num_active_jobs
    all_policies = args.policies
    oracle_throughputs = utils.read_all_throughputs_json(args.throughputs_file)

    print('Cluster spec,# Active Jobs,Policy,Trials,Runtime,Stddev')
    for cluster_spec in all_cluster_specs:
        for num_active_jobs in all_num_active_jobs:
            for policy in all_policies:
                measure_runtime(cluster_spec, num_active_jobs, policy,
                                oracle_throughputs,
                                args.generate_multi_gpu_jobs,
                                args.generate_multi_priority_jobs,
                                args.num_trials)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--throughputs-file', type=str,
                        default='oracle_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('-c', '--cluster-spec', type=str, nargs='+',
                        default=['64:0:0', '36:36:36', '1500:0:0',
                                 '850:850:850'],
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('-n', '--num_active_jobs', type=int, nargs='+',
                        default=[2**i for i in range(4, 10)],
                        help='List of number of active jobs to sweep')
    parser.add_argument('-p', '--policies', type=str, nargs='+',
                        default=['max_min_fairness_packed'],
                        help='List of policies to sweep')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials to run for each experiment')
    args = parser.parse_args()

    main(args)
