import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import numpy as np
import random
import time

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable

def generate_job(rng, oracle_throughputs, generate_multi_gpu_jobs=False,
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

def get_job_throughputs(jobs, oracle_throughputs, worker_types):
    throughputs = {}
    for i, job in enumerate(jobs):
        throughputs[job.job_id] = {}
        for worker_type in worker_types:
            job_type_key = (job.job_type, job.scale_factor)
            throughputs[job.job_id][worker_type] = \
                oracle_throughputs[worker_type][job_type_key]['null']
        for j, other_job in enumerate(jobs[i+1:]):
            merged_job_id = JobIdPair(job.job_id[0], other_job.job_id[0])
            if other_job.scale_factor != job.scale_factor:
                continue
            throughputs[merged_job_id] = {}
            other_job_type_key = (other_job.job_type, other_job.scale_factor)
            for worker_type in worker_types:
                throughputs[merged_job_id][worker_type] = \
                    oracle_throughputs[worker_type][job_type_key][other_job_type_key]
    return throughputs

def get_job_type_throughputs(jobs, oracle_throughputs, worker_types):
    job_type_keys = set([(job.job_type, job.scale_factor) for job in jobs])
    job_type_throughputs = {}
    for job_type_key in job_type_keys:
        job_type_throughputs[job_type_key] = {}
        for worker_type in worker_types:
            job_type_throughputs[job_type_key][worker_type] = {}
            job_type_throughputs[job_type_key][worker_type][None] = \
                oracle_throughputs[worker_type][job_type_key]['null']
            for other_job_type_key in job_type_keys:
                if other_job_type_key[1] != job_type_key[1]:
                    continue
                job_type_throughputs[job_type_key][worker_type][other_job_type_key] = \
                    oracle_throughputs[worker_type][job_type_key][other_job_type_key][0]
    return job_type_throughputs

def get_allocation(policy, jobs, oracle_throughputs, cluster_spec,
                   worker_types, scale_factors, priority_weights):
    throughputs = get_job_throughputs(jobs, oracle_throughputs, worker_types)
    unflattened_allocation = \
        policy.get_allocation(throughputs, scale_factors, priority_weights,
                              cluster_spec)
    return unflattened_allocation


def get_allocation_using_job_type_throughputs(policy, jobs,
                                              oracle_throughputs, cluster_spec,
                                              worker_types, scale_factors,
                                              priority_weights):
    job_id_to_job_type_key = \
        {job.job_id : (job.job_type, job.scale_factor) for job in jobs}
    job_type_throughputs = get_job_type_throughputs(jobs, oracle_throughputs,
                                                    worker_types)
    unflattened_allocation = \
        policy.get_allocation_using_job_type_throughputs(
                job_type_throughputs, job_id_to_job_type_key, scale_factors,
                priority_weights, cluster_spec)
    return unflattened_allocation


def print_effective_throughputs(allocation, throughputs,
                                job_id_to_job_type_key):
    """Prints effective throughputs given a job-job allocation."""
    job_ids = sorted(allocation.keys())
    worker_types = sorted(allocation[job_ids[0]].keys())

    relevant_job_ids = {}
    for job_id in job_ids:
        for single_job_id in job_id.singletons():
            if single_job_id not in relevant_job_ids:
                relevant_job_ids[single_job_id] = []
            relevant_job_ids[single_job_id].append(job_id)

    single_job_ids = sorted(relevant_job_ids.keys())

    normalizing_factors = {}
    for single_job_id in single_job_ids:
        normalizing_factors[single_job_id] = 0.0
        job_type_key = job_id_to_job_type_key[single_job_id]
        for worker_type in worker_types:
            normalizing_factors[single_job_id] += \
                throughputs[worker_type][job_type_key]['null']

    effective_throughputs = {}
    for single_job_id in single_job_ids:
        job_type_key = job_id_to_job_type_key[single_job_id]
        effective_throughput = 0.0
        for job_id in relevant_job_ids[single_job_id]:
            for worker_type in worker_types:
                if not job_id.is_pair():
                    throughput = throughputs[worker_type][job_type_key]['null']
                else:
                    index = job_id.singletons().index(single_job_id)
                    other_job_id = \
                        job_id.singletons()[1 - index]
                    other_job_type_key = \
                        job_id_to_job_type_key[other_job_id]
                    if other_job_type_key[1] != job_type_key[1]:
                        continue
                    throughput = \
                        throughputs[worker_type][job_type_key][other_job_type_key][0]
                throughput /= normalizing_factors[single_job_id]
                effective_throughput += \
                    allocation[job_id][worker_type] * throughput
        print('%s: %f' % (single_job_id, effective_throughput))

def main(args):
    rng = random.Random()
    rng.seed(0)
    v100s, p100s, k80s = args.cluster_spec.split(':')
    cluster_spec = {
        'v100': int(v100s),
        'p100': int(p100s),
        'k80': int(k80s),
    }
    worker_types = sorted(cluster_spec.keys())
    oracle_throughputs =\
        utils.read_all_throughputs_json_v2(args.throughputs_file)
    jobs = []
    for i in range(args.num_jobs):
        job_template = JobTable[i % len(JobTable)]
        job_id = JobIdPair(i, None)
        jobs.append(generate_job(rng, oracle_throughputs,
                                 args.generate_multi_gpu_jobs,
                                 args.generate_multi_priority_jobs,
                                 job_template=job_template,
                                 job_id=job_id))
    policy = utils.get_policy('max_min_fairness_packed', solver=args.solver)
    scale_factors = {
        job.job_id: job.scale_factor for job in jobs
    }
    priority_weights = {
        job.job_id: job.priority_weight for job in jobs
    }

    start = datetime.datetime.now()
    original_allocation = get_allocation(policy, jobs, oracle_throughputs,
                                         cluster_spec, worker_types,
                                         scale_factors, priority_weights)
    original_runtime = datetime.datetime.now() - start

    start = datetime.datetime.now()
    job_type_allocation = \
        get_allocation_using_job_type_throughputs(policy, jobs,
                                                  oracle_throughputs,
                                                  cluster_spec, worker_types,
                                                  scale_factors,
                                                  priority_weights)
    job_id_to_job_type_key = \
        {job.job_id: (job.job_type, job.scale_factor) for job in jobs}
    job_type_runtime = datetime.datetime.now() - start

    if args.verbose:
        print('Original allocation:')
        utils.print_allocation(original_allocation)
        print('')

        print('Allocation using job type throughputs:')
        utils.print_allocation(job_type_allocation)
        print('')

    print('Original effective throughputs:')
    print_effective_throughputs(original_allocation, oracle_throughputs,
                                job_id_to_job_type_key)
    print('')

    print('Effective_throughputs using job type throughputs:')
    print_effective_throughputs(job_type_allocation, oracle_throughputs,
                                job_id_to_job_type_key)

    print('Original runtime:',
          original_runtime.seconds + original_runtime.microseconds / 1.0e6)
    print('Runtime using job type throughputs:',
          job_type_runtime.seconds + job_type_runtime.microseconds / 1.0e6)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--num_jobs', type=int, default=10, help='Num jobs')
    parser.add_argument('--throughputs-file', type=str,
                        default=('oracle_throughputs_v2.json'),
                        help='Oracle throughputs file')
    parser.add_argument('--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('-c', '--cluster-spec', type=str, default='3:3:3',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose')
    args = parser.parse_args()
    main(args)
