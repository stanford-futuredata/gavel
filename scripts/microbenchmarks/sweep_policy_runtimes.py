import argparse
import random
import time

import sys; sys.path.append(".")

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable


def _generate_job(oracle_throughputs, generate_multi_gpu_jobs=False,
                  generate_multi_priority_jobs=False,
                  run_dir='/tmp'):
    """Generates a new job for simulation."""
    job_template = random.choice(JobTable)
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

def measure_runtime(policy_name, all_num_active_jobs, cluster_specs,
                    oracle_throughputs, generate_multi_gpu_jobs,
                    generate_multi_priority_jobs):
    policy = utils.get_policy(policy_name)
    for cluster_spec in cluster_specs:
        for num_active_jobs in all_num_active_jobs:
            if num_active_jobs > 512 and "_Packing" in policy.name:
                continue
            throughputs = {}
            jobs = {}
            for i in range(num_active_jobs):
                job_id = JobIdPair(i, None)
                jobs[i] = _generate_job(oracle_throughputs,
                                        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                                        generate_multi_priority_jobs=generate_multi_priority_jobs)
                throughputs[job_id] = {}
                for worker_type in cluster_spec:
                    throughputs[job_id][worker_type] = \
                        oracle_throughputs[worker_type][jobs[i].job_type]["null"]
            if "_Packing" in policy.name:
                for i in range(num_active_jobs):
                    for j in range(num_active_jobs):
                        if i != j and jobs[i].scale_factor == jobs[j].scale_factor:
                            throughputs[JobIdPair(i, j)] = {}
                            for worker_type in cluster_spec:
                                throughputs[JobIdPair(i, j)][worker_type] = \
                                    oracle_throughputs[
                                        worker_type][jobs[i].job_type][jobs[j].job_type]
            scale_factors = {
                JobIdPair(i, None): jobs[i].scale_factor for i in range(num_active_jobs)
            }
            start_time = time.time()
            if policy.name.startswith("MaxMinFairness"):
                priority_weights = {
                    JobIdPair(i, None): jobs[i].priority_weight for i in range(num_active_jobs)
                }
                policy.get_allocation(
                    throughputs, scale_factors, priority_weights,
                    cluster_spec)
            elif policy.name.startswith("MinTotalDuration"):
                num_steps_remaining = {
                    JobIdPair(i, None): jobs[i].num_steps for i in range(num_active_jobs)
                }
                policy.get_allocation(
                    throughputs, scale_factors, num_steps_remaining,
                    cluster_spec)
            else:
                policy.get_allocation(
                    throughputs, scale_factors, cluster_spec)
            print("[Policy=%s, Number of active jobs=%d, Cluster_spec=%s] %.3f seconds" % (
                policy_name, num_active_jobs, cluster_spec,
                time.time() - start_time))


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
    args = parser.parse_args()

    all_num_active_jobs = [2**i for i in range(4, 15)]
    cluster_specs = [{"v100": 64, "p100": 0, "k80": 0},
                     {"v100": 36, "p100": 36, "k80": 36},
                     {"v100": 1500, "p100": 0, "k80": 0},
                     {"v100": 850, "p100": 850, "k80": 850}]
    oracle_throughputs = utils.read_all_throughputs_json(args.throughputs_file)
    for policy_name in ['max_min_fairness', 'max_min_fairness_perf', 'max_min_fairness_packed']:
        measure_runtime(policy_name, all_num_active_jobs, cluster_specs,
                        oracle_throughputs, args.generate_multi_gpu_jobs,
                        args.generate_multi_priority_jobs)
