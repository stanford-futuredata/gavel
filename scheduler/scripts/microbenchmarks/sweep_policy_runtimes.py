import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import contextlib
import numpy as np
import random
import time

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable

def _generate_job(rng, oracle_throughputs, generate_multi_gpu_jobs=False,
                  generate_multi_priority_jobs=False,
                  run_dir='/tmp', job_template=None, job_id=None):
    """Generates a new job for simulation."""

    scale_factor = 1
    if ((job_template is not None and job_template.distributed and
         generate_multi_gpu_jobs) or
        (job_template is None and generate_multi_gpu_jobs)):
        r = rng.uniform(0, 1)
        if 0.7 <= r <= 0.8:
            scale_factor = 2
        elif 0.8 <= r <= 0.95:
            scale_factor = 4
        elif 0.95 <= r:
            scale_factor = 8
    if job_template is None:
        while True:
            job_template = rng.choice(JobTable)
            if scale_factor == 1 or job_template.distributed:
                break
    job_type = job_template.model
    run_time = 60 * (10 ** random.uniform(2, 4))
    assert(run_time > 0)
    num_steps = \
        run_time * oracle_throughputs['v100'][(job_type, scale_factor)]['null']
    assert(num_steps > 0)

    if job_template.needs_data_dir:
        command = job_template.command % (run_dir, run_dir)
    else:
        command = job_template.command % (run_dir)

    priority_weight = 1.0
    if generate_multi_priority_jobs:
        # NOTE: Distribution modified for test purposes.
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.4:
            priority_weight = 5.0

    job = Job(job_id=job_id,
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
                    generate_multi_priority_jobs, num_trials, solver):
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
        job_type_key = (jobs[i].job_type, jobs[i].scale_factor)
        throughputs[job_id] = {}
        for worker_type in cluster_spec:
            throughputs[job_id][worker_type] = \
                oracle_throughputs[worker_type][job_type_key]['null']
        if 'pack' in policy_name:
            for j in range(num_active_jobs):
                if i < j and jobs[i].scale_factor == jobs[j].scale_factor:
                    other_job_type_key = \
                        (jobs[j].job_type, jobs[j].scale_factor)
                    throughputs[JobIdPair(i, j)] = {}
                    for worker_type in cluster_spec:
                        throughputs[JobIdPair(i, j)][worker_type] = \
                            oracle_throughputs[worker_type][job_type_key][other_job_type_key]
    scale_factors = {
        JobIdPair(i, None): jobs[i].scale_factor for i in range(num_active_jobs)
    }
    results_str = '%s,%d' % (policy_name, num_active_jobs)
    results = []
    for trial in range(num_trials):
        policy = utils.get_policy(policy_name, solver=solver)
        start_time = time.time()
        with open('/dev/null', 'w') as f:
            with contextlib.redirect_stdout(f):
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
    for result in results:
        results_str += ',' + str(result)
    results_str += ',' + str(np.mean(results))
    return results_str

def main(args):
    cluster_spec = args.cluster_spec
    all_num_active_jobs = args.num_active_jobs
    all_policies = args.policies
    oracle_throughputs =\
        utils.read_all_throughputs_json_v2(args.throughputs_file)

    if args.output_file is not None:
        output_file = open(args.output_file, 'w')
    else:
        output_file = None

    header_str = 'Policy,# Jobs'
    for i in range(args.num_trials):
        header_str += ',Trial %d' % (i+1)
    header_str += ',Average'
    if output_file is not None:
        output_file.write('%s\n' % (header_str))
    print(header_str)

    for policy in all_policies:
        for num_active_jobs in all_num_active_jobs:
            results = measure_runtime(cluster_spec, num_active_jobs,
                                      policy, oracle_throughputs,
                                      args.generate_multi_gpu_jobs,
                                      args.generate_multi_priority_jobs,
                                      args.num_trials, args.solver)
            if output_file is not None:
                output_file.write('%s\n' % (results))
            print(results)

    if output_file is not None:
        output_file.close

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--throughputs-file', type=str,
                        default='oracle_throughputs_v2.json',
                        help='Oracle throughputs file')
    parser.add_argument('--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('-c', '--cluster-spec', type=str, default='36:36:36',
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
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default=None, help='CVXPY solver')
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to output results to')
    args = parser.parse_args()

    main(args)
