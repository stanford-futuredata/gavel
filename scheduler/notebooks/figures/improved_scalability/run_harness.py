import sys; sys.path.append("../../..")
from job_id_pair import JobIdPair
import utils

import copy
import numpy as np
import random
import time

np.set_printoptions(precision=3, suppress=True)


def create_problem_instance(num_jobs, cluster_spec,
                            policy_name,
                            seed,
                            introduce_skew=False):
    oracle_throughputs = utils.read_all_throughputs_json_v2("../../../simulation_throughputs.json")
    rng = random.Random()
    rng.seed(seed)
    jobs = {}
    throughputs = {}
    scale_factors = {}
    priority_weights = {}
    for i in range(num_jobs):
        job_id = JobIdPair(i, None)
        job = utils.generate_job(throughputs=oracle_throughputs,
                                 rng=rng, job_id=job_id)
        jobs[job_id[0]] = job
        job_type_key = (job.job_type, job.scale_factor)
        throughputs[job_id] = {}
        for worker_type in cluster_spec:
            throughputs[job_id][worker_type] = \
                oracle_throughputs[worker_type][job_type_key]['null']
        scale_factors[job_id] = 1
        if introduce_skew:
            priority_weights[job_id] = (i % 4) + 1.0
        else:
            priority_weights[job_id] = 1.0
    if 'pack' in policy_name:
        for i in range(num_jobs):
            job_type_key = (jobs[i].job_type, jobs[i].scale_factor)
            for j in range(num_jobs):
                if i < j and jobs[i].scale_factor == jobs[j].scale_factor:
                    other_job_type_key = \
                        (jobs[j].job_type, jobs[j].scale_factor)
                    throughputs[JobIdPair(i, j)] = {}
                    for worker_type in cluster_spec:
                        throughputs[JobIdPair(i, j)][worker_type] = \
                            oracle_throughputs[worker_type][job_type_key][other_job_type_key]
    return throughputs, scale_factors, priority_weights


def harness(policy, throughputs, scale_factors, priority_weights, cluster_spec, num_sub_clusters=1,
            random_cluster_assignment=False):
    start_time = time.time()
    sub_cluster_throughputs = []
    sub_cluster_scale_factors = []
    sub_cluster_priority_weights = []
    job_to_sub_cluster_assignment = {}
    job_ids = []
    for job_id in throughputs:
        if not job_id.is_pair():
            job_ids.append(job_id)
    for i, job_id in enumerate(job_ids):
        if random_cluster_assignment:
            job_to_sub_cluster_assignment[job_id[0]] = random.randint(0, num_sub_clusters-1)
        else:
            job_to_sub_cluster_assignment[job_id[0]] = job_id[0] % num_sub_clusters
    for i in range(num_sub_clusters):
        sub_cluster_throughputs.append({})
        sub_cluster_scale_factors.append({})
        sub_cluster_priority_weights.append({})
        for job_id in throughputs:
            if (job_to_sub_cluster_assignment[job_id[0]] == i) and (
                 not job_id.is_pair() or (job_to_sub_cluster_assignment[job_id[1]] == i)):
                sub_cluster_throughputs[-1][job_id] = copy.copy(throughputs[job_id])
                if not job_id.is_pair():
                    sub_cluster_scale_factors[-1][job_id] = scale_factors[job_id]
                    sub_cluster_priority_weights[-1][job_id] = priority_weights[job_id]
    sub_cluster_cluster_spec = {worker_type: cluster_spec[worker_type] // num_sub_clusters
                                for worker_type in cluster_spec}
    setup_time = time.time() - start_time
    full_allocation = {}
    computation_times = []
    for i in range(num_sub_clusters):
        start_time = time.time()
        if policy._name.startswith('MaxMinFairness'):
            sub_cluster_allocation = policy.get_allocation(
                sub_cluster_throughputs[i], sub_cluster_scale_factors[i],
                sub_cluster_priority_weights[i], sub_cluster_cluster_spec)
        else:
            sub_cluster_allocation = policy.get_allocation(
                sub_cluster_throughputs[i], sub_cluster_scale_factors[i],
                sub_cluster_cluster_spec)
        for job_id in sub_cluster_allocation:
            full_allocation[job_id] = sub_cluster_allocation[job_id]
        computation_times.append(time.time() - start_time)
    return full_allocation, setup_time + max(computation_times)


def sweep(policy_names_and_num_sub_clusters,
          all_num_jobs,
          num_trials, introduce_skew=False):
    all_runtimes = {}
    all_effective_throughputs = {}
    for num_jobs in all_num_jobs:
        all_runtimes[num_jobs] = []
        all_effective_throughputs[num_jobs] = []
        cluster_spec = {
            'v100': max(num_jobs // 4, 1),
            'p100': max(num_jobs // 4, 1),
            'k80': max(num_jobs // 4, 1),
        }
        for i in range(num_trials):
            throughputs, scale_factors, priority_weights = \
                create_problem_instance(num_jobs, cluster_spec,
                                        policy_names_and_num_sub_clusters[0][0], seed=i,
                                        introduce_skew=introduce_skew)
            all_runtimes[num_jobs].append([])
            allocations = []
            for (policy_name, num_sub_clusters) in policy_names_and_num_sub_clusters:
                policy = utils.get_policy(policy_name, solver='ECOS')
                allocation, runtime = harness(
                    policy, throughputs,
                    scale_factors,
                    priority_weights,
                    cluster_spec,
                    num_sub_clusters=num_sub_clusters)
                all_runtimes[num_jobs][-1].append(runtime)
                allocations.append(allocation)

            all_effective_throughputs[num_jobs].append([])
            for allocation in allocations:
                effective_throughputs = {}
                for job_id in allocation:
                    for single_job_id in job_id.singletons():
                        if single_job_id not in effective_throughputs:
                            effective_throughputs[single_job_id] = 0.0
                    for worker_type in allocation[job_id]:
                        if job_id.is_pair():
                            for i, single_job_id in enumerate(job_id.singletons()):
                                effective_throughputs[single_job_id] += (
                                   allocation[job_id][worker_type] *
                                   throughputs[job_id][worker_type][i]
                                )
                        else:
                            effective_throughputs[job_id] += (
                                allocation[job_id][worker_type] *
                                throughputs[job_id][worker_type])
                all_effective_throughputs[num_jobs][-1].append(effective_throughputs)
    return all_runtimes, all_effective_throughputs


def get_runtimes_and_effective_throughputs(policy_name, all_num_sub_clusters,
                                           num_jobs, introduce_skew=False):
    policy_names_and_num_sub_clusters = [
        (policy_name, num_sub_clusters)
        for num_sub_clusters in all_num_sub_clusters
    ]
    if not introduce_skew:
        policy_names_and_num_sub_clusters.append(
            ('gandiva', 1))
    all_runtimes, all_effective_throughputs = sweep(
        policy_names_and_num_sub_clusters, [num_jobs],
        num_trials=1, introduce_skew=introduce_skew)
    runtimes = all_runtimes[num_jobs][0]
    all_effective_throughputs = all_effective_throughputs[num_jobs][0]
    return runtimes, all_effective_throughputs


if __name__ == '__main__':
    all_num_sub_clusters = [1, 2, 4, 8]
    runtimes, all_effective_throughputs = \
        get_runtimes_and_effective_throughputs('max_min_fairness_packed',
                                               all_num_sub_clusters,
                                               1024)
    print(runtimes)
    print(all_effective_throughputs)
