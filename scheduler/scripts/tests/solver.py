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

def generate_job(rng, oracle_throughputs, generate_multi_gpu_jobs=False,
                  generate_multi_priority_jobs=False,
                  run_dir='/tmp', job_template=None, job_id=None):
    """Generates a new job for simulation."""
    if job_template is None:
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
            throughputs[job.job_id][worker_type] = \
                oracle_throughputs[worker_type][job.job_type]['null']
        for j, other_job in enumerate(jobs[i+1:]):
            merged_job_id = JobIdPair(job.job_id[0], other_job.job_id[0])
            throughputs[merged_job_id] = {}
            for worker_type in worker_types:
                throughputs[merged_job_id][worker_type] = \
                    oracle_throughputs[worker_type][job.job_type][other_job.job_type]
    return throughputs

def get_app_throughputs(jobs, oracle_throughputs, worker_types):
    apps = set([job.job_type for job in jobs])
    app_throughputs = {}
    for app in apps:
        app_throughputs[app] = {}
        for worker_type in worker_types:
            app_throughputs[app][worker_type] = {}
            app_throughputs[app][worker_type][None] = \
                oracle_throughputs[worker_type][app]['null']
            for other_app in apps:
                app_throughputs[app][worker_type][other_app] = \
                    oracle_throughputs[worker_type][app][other_app][0]
    return app_throughputs

def get_allocation_v1(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types):
    throughputs = get_job_throughputs(jobs, oracle_throughputs, worker_types)
    scale_factors = {}
    for job in jobs:
        scale_factors[job.job_id] = 1
    unflattened_allocation = \
        policy.get_allocation(throughputs, scale_factors, None, cluster_spec)
    n = len(jobs)
    apps = sorted(set([job.job_type for job in jobs]))
    num_variables_per_job = 1 + len(apps)
    flattened_allocation = np.zeros((n * num_variables_per_job,
                                     len(worker_types)), dtype=np.float32)
    job_ids = sorted([job.job_id for job in jobs])
    for i, job_id in enumerate(job_ids):
        job_offset = i * num_variables_per_job
        for k, worker_type in enumerate(worker_types):
            flattened_allocation[job_offset,k] = \
                unflattened_allocation[job_id][worker_type]
        for j, other_job_id in enumerate(job_ids):
            if i == j:
                continue
            merged_job_id = JobIdPair(job_id[0], other_job_id[0])
            app_idx = apps.index(jobs[j].job_type)
            for k, worker_type in enumerate(worker_types):
                flattened_allocation[job_offset+1+app_idx, k] += \
                    unflattened_allocation[merged_job_id][worker_type]
    print('Flattened v1 allocation:')
    print(flattened_allocation.round(5))
    print('')

    return unflattened_allocation

def get_allocation_v2(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types):
    job_id_to_application = {job.job_id : job.job_type for job in jobs}
    app_throughputs = get_app_throughputs(jobs, oracle_throughputs,
                                          worker_types)
    apps = sorted(app_throughputs.keys())
    num_variables_per_job = len(app_throughputs.keys()) + 1
    flattened_allocation = policy.get_allocation_v2(app_throughputs,
                                                    job_id_to_application,
                                                    None, None, cluster_spec)
    print('Flattened v2 allocation:')
    print(flattened_allocation.round(5))
    unflattened_allocation = {}
    for i, job in enumerate(jobs):
        unflattened_allocation[job.job_id] = {}
        for j, worker_type in enumerate(worker_types):
            unflattened_allocation[job.job_id][worker_type] = \
                flattened_allocation[i*num_variables_per_job,j]
        for k, other_job in enumerate(jobs):
            if i == k:
                continue
            merged_job_id = JobIdPair(job.job_id[0], other_job.job_id[0])
            if merged_job_id not in unflattened_allocation:
                unflattened_allocation[merged_job_id] = {}
            for j, worker_type in enumerate(worker_types):
                app_idx = apps.index(other_job.job_type)
                if worker_type not in unflattened_allocation[merged_job_id]:
                    unflattened_allocation[merged_job_id][worker_type] = 0
                unflattened_allocation[merged_job_id][worker_type] += \
                    flattened_allocation[i*num_variables_per_job+1+app_idx,j]

    return unflattened_allocation

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
    oracle_throughputs = utils.read_all_throughputs_json(args.throughputs_file)
    jobs = []
    for i in range(args.num_jobs):
        job_template = JobTable[i % len(JobTable)]
        job_id = JobIdPair(i, None)
        jobs.append(generate_job(rng, oracle_throughputs,
                                 args.generate_multi_gpu_jobs,
                                 args.generate_multi_priority_jobs,
                                 job_template=job_template,
                                 job_id=job_id))
    policy = utils.get_policy('max_min_fairness_packed')
    allocation_v1 = get_allocation_v1(policy, jobs, oracle_throughputs,
                                      cluster_spec, worker_types)
    allocation_v2 = get_allocation_v2(policy, jobs, oracle_throughputs,
                                      cluster_spec, worker_types)

    print('Allocation v1:')
    utils.print_allocation(allocation_v1)
    print('Allocation v2:')
    utils.print_allocation(allocation_v2)

    """
    job_ids = sorted(allocation_v2.keys())
    for job_id in job_ids:
        print('Job', str(job_id) + ':', allocation_v2[job_id])
        print('')
    """

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--num_jobs', type=int, default=10, help='Num jobs')
    parser.add_argument('--throughputs-file', type=str,
                        default=('oracle_throughputs.json'),
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
    args = parser.parse_args()
    main(args)

