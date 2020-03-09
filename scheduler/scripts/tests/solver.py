import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import numpy as np
import random
from scipy.optimize import nnls, lsq_linear
import time

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable

def convert_job_type_allocation_v2(allocation, job_id_to_job_type):
    """Converts a job-job_type allocation to a job-job allocation."""
    job_ids = sorted(allocation.keys())
    worker_types = sorted(allocation[job_ids[0]].keys())
    job_types = sorted(set([job_id_to_job_type[job_id] for job_id in job_ids]))

    # Initialize job_type-job_type allocation.
    job_type_allocation = {}
    for worker_type in worker_types:
        job_type_allocation[worker_type] = {}
        for job_type in job_types:
            job_type_allocation[worker_type][job_type] = {}
            job_type_allocation_ = job_type_allocation[worker_type][job_type]
            for other_job_type in [None] + job_types:
                job_type_allocation_[other_job_type] = 0.0

    # Populate job_type-job_type allocation.
    for worker_type in worker_types:
        for job_id in allocation:
            job_type = job_id_to_job_type[job_id]
            for other_job_type in allocation[job_id][worker_type]:
                job_type_allocation[worker_type][job_type][other_job_type] += \
                    allocation[job_id][worker_type][other_job_type]

    # Compute job-job allocations using the following formula:
    # x_{i,j} = x_{i, job_type(j)} * x_{j, job_type(i)} /
    #   sum x_{k, job_type(j)} for all k of job_type(i)
    converted_allocation = {}
    for i, job_id in enumerate(job_ids):
        converted_allocation[job_id] = {}
        job_type = job_id_to_job_type[job_id]
        for worker_type in worker_types:
            converted_allocation[job_id][worker_type] = \
                allocation[job_id][worker_type][None]
        for other_job_id in job_ids[i+1:]:
            other_job_type = job_id_to_job_type[other_job_id]
            merged_job_id = JobIdPair(job_id[0], other_job_id[0])
            converted_allocation[merged_job_id] = {}
            for worker_type in worker_types:
                current_job_type_allocation = \
                    job_type_allocation[worker_type][job_type][other_job_type]
                if current_job_type_allocation > 0.0:
                    # TODO: Remove this when we have a correct solution.
                    if job_type == other_job_type:
                        current_job_type_allocation -= \
                            allocation[job_id][worker_type][job_type]
                    converted_allocation[merged_job_id][worker_type] = \
                        (allocation[job_id][worker_type][other_job_type] *\
                         allocation[other_job_id][worker_type][job_type] /\
                         current_job_type_allocation)
                else:
                    converted_allocation[merged_job_id][worker_type] = 0.0

    return converted_allocation

def convert_job_type_allocation(allocation, job_id_to_job_type):
    job_ids = sorted(allocation.keys())
    job_types = sorted(set([job_id_to_job_type[job_id] for job_id in job_ids]))
    worker_types = sorted(allocation[job_ids[0]].keys())
    converted_allocation = {}

    for worker_type in worker_types:
        v1_vars = []
        v2_vars = []
        v2_var_to_v1_vars = {}

        for i, job_id in enumerate(job_ids):
            for job_type in [''] + job_types:
                v2_vars.append((job_id, job_type))
                v2_var_to_v1_vars[v2_vars[-1]] = []

        for i, job_id in enumerate(job_ids):
            job_type = job_id_to_job_type[job_id]
            v1_vars.append(job_id)
            v2_var_to_v1_vars[(job_id, '')].append(v1_vars[-1])
            for other_job_id in job_ids[i+1:]:
                colocated_job_type = job_id_to_job_type[other_job_id]
                v1_vars.append(JobIdPair(job_id[0], other_job_id[0]))
                if (job_id, colocated_job_type) in v2_var_to_v1_vars:
                    v2_var = (job_id, colocated_job_type)
                    v2_var_to_v1_vars[v2_var].append(v1_vars[-1])
                if (other_job_id, job_type) in v2_var_to_v1_vars:
                    v2_var = (other_job_id, job_type)
                    v2_var_to_v1_vars[v2_var].append(v1_vars[-1])

        # Filter out all variables that have 0 allocation.
        zero_v2_vars = set()
        zero_v1_vars = set()
        for i, v2_var in enumerate(v2_vars):
            (job_id, colocated_job_type) = v2_var
            # NOTE: We have to store the "none" colocated job type as an empty
            # string to enable taking set differences. So we convert the empty
            # string back to None here.
            if colocated_job_type == '':
                colocated_job_type = None
            if allocation[job_id][worker_type][colocated_job_type] <= 1e-10:
                zero_v2_vars.add(v2_var)
                for v1_var in v2_var_to_v1_vars[v2_var]:
                    if v1_var not in converted_allocation:
                        converted_allocation[v1_var] = {}
                    converted_allocation[v1_var][worker_type] = 0
                    zero_v1_vars.add(v1_var)

        print('%d/%d (%.2f) of job-job type allocation '
              'values are 0' % (len(zero_v2_vars), len(v2_vars),
                                float(len(zero_v2_vars)) / len(v2_vars) * 100))

        v1_vars = sorted(set(v1_vars) - zero_v1_vars)
        v2_vars = sorted(set(v2_vars) - zero_v2_vars)

        # Build map from v1_var to index.
        index = {}
        for i, v1_var in enumerate(v1_vars):
            index[v1_var] = i

        a = np.zeros((len(v2_vars), len(v1_vars)))
        b = np.zeros(len(v2_vars))
        for i, v2_var in enumerate(v2_vars):
            (job_id, colocated_job_type) = v2_var
            # NOTE: We have to store the "none" colocated job type as an empty
            # string to enable taking set differences. So we convert the empty
            # string back to None here.
            if colocated_job_type == '':
                colocated_job_type = None
            for v1_var in v2_var_to_v1_vars[v2_var]:
                if v1_var in v1_vars:
                    j = index[v1_var]
                    a[i,j] = 1
            b[i] = allocation[job_id][worker_type][colocated_job_type]

        result = lsq_linear(a, b, bounds=(0, 1), method='trf',
                            lsq_solver='lsmr', lsmr_tol='auto',
                            verbose=2)

        for i, v1_var in enumerate(v1_vars):
            if v1_var not in converted_allocation:
                converted_allocation[v1_var] = {}
            converted_allocation[v1_var][worker_type] = result.x[i]
    return converted_allocation

def print_unflattened_v1_allocation(allocation):
    for job_id in sorted(allocation.keys()):
        print('%s: %.5f %.5f %.5f' % (job_id, allocation[job_id]['k80'],
                                      allocation[job_id]['p100'],
                                      allocation[job_id]['v100']))

def print_allocation(allocation, jobs, v3=False):
    n = len(jobs)
    job_types = sorted(set([job.job_type for job in jobs]))
    k = 1 + len(job_types)
    job_ids = sorted([job.job_id for job in jobs])
    job_id_to_job_type = {job.job_id : job.job_type for job in jobs}
    if v3:
        for i, job_id in enumerate(job_ids):
            print('Job %s (%s):' % (str(job_id),
                                    job_id_to_job_type[job_id]))
            for j, job_type in enumerate([None] + job_types):
                if j == 0:
                    print('\tIsolated allocation: '
                          '%.5f %.5f %.5f' % (allocation[i,0],
                                              allocation[i,k],
                                              allocation[i,2*k]))
                else:
                    print('\tAllocation with %s: '
                          '%.5f %.5f %.5f' % (job_type,
                                              allocation[i,j],
                                              allocation[i,k+j],
                                              allocation[i,2*k+j]))

    else:
        for i in range(0, n * k, k):
            job_id = job_ids[i // k]
            print('Job %s (%s):' % (str(job_id),
                                    job_id_to_job_type[job_id]))
            print('\tIsolated allocation: '
                  '%.5f %.5f %.5f' % (allocation[i,0],
                                      allocation[i,1],
                                      allocation[i,2]))
            for j in range(1, k):
                print('\tAllocation with %s: '
                      '%.5f %.5f %.5f' % (job_types[j-1],
                                          allocation[i+j,0],
                                          allocation[i+j,1],
                                          allocation[i+j,2]))


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
    if generate_multi_gpu_jobs:
        # NOTE: Distribution modified for test purposes.
        r = rng.uniform(0, 1)
        if 0.2 <= r <= 0.6:
            scale_factor = 2
        elif 0.6 <= r <= 0.9:
            scale_factor = 4
        elif 0.9 <= r:
            scale_factor = 8

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
            throughputs[job.job_id][worker_type] = \
                oracle_throughputs[worker_type][job.job_type]['null']
        for j, other_job in enumerate(jobs[i+1:]):
            merged_job_id = JobIdPair(job.job_id[0], other_job.job_id[0])
            throughputs[merged_job_id] = {}
            for worker_type in worker_types:
                throughputs[merged_job_id][worker_type] = \
                    oracle_throughputs[worker_type][job.job_type][other_job.job_type]
    return throughputs

def get_job_type_throughputs(jobs, oracle_throughputs, worker_types):
    job_types = set([job.job_type for job in jobs])
    job_type_throughputs = {}
    for job_type in job_types:
        job_type_throughputs[job_type] = {}
        for worker_type in worker_types:
            job_type_throughputs[job_type][worker_type] = {}
            job_type_throughputs[job_type][worker_type][None] = \
                oracle_throughputs[worker_type][job_type]['null']
            for other_job_type in job_types:
                job_type_throughputs[job_type][worker_type][other_job_type] = \
                    oracle_throughputs[worker_type][job_type][other_job_type][0]
    return job_type_throughputs

def get_allocation_v1(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types, scale_factors, priority_weights,
                      flatten=True):
    throughputs = get_job_throughputs(jobs, oracle_throughputs, worker_types)
    unflattened_allocation = \
        policy.get_allocation(throughputs, scale_factors, priority_weights,
                              cluster_spec)
    if not flatten:
        return unflattened_allocation

    n = len(jobs)
    job_types = sorted(set([job.job_type for job in jobs]))
    num_variables_per_job = 1 + len(job_types)
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
            job_type_idx = job_types.index(jobs[j].job_type)
            for k, worker_type in enumerate(worker_types):
                flattened_allocation[job_offset+1+job_type_idx, k] = \
                    unflattened_allocation[merged_job_id][worker_type]
    return flattened_allocation

def get_allocation_v2(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types, scale_factors, priority_weights,
                      flatten=True):
    job_id_to_job_type = {job.job_id : job.job_type for job in jobs}
    job_type_throughputs = get_job_type_throughputs(jobs, oracle_throughputs,
                                          worker_types)
    flattened_allocation = policy.get_allocation_v2(job_type_throughputs,
                                                    job_id_to_job_type,
                                                    scale_factors,
                                                    priority_weights,
                                                    cluster_spec)
    if flatten:
        return flattened_allocation

    # TODO: Unflatten allocation
    unflattened_allocation = {}
    return unflattned_allocation

def get_allocation_v3(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types, scale_factors, priority_weights,
                      flatten=False):
    job_id_to_job_type = {job.job_id : job.job_type for job in jobs}
    job_type_throughputs = get_job_type_throughputs(jobs, oracle_throughputs,
                                          worker_types)
    unflattened_allocation = policy.get_allocation_v3(job_type_throughputs,
                                                      job_id_to_job_type,
                                                      scale_factors,
                                                      priority_weights,
                                                      cluster_spec)
    if flatten:
        # TODO: Flatten allocation.

        # Num jobs.
        n = len(jobs)
        # Num job_types.
        a = len(job_type_throughputs.keys())
        # Num worker_types.
        m = len(worker_types)
        # Num varibles per job.
        num_vars_per_job = 1 + a

        flattened_allocation = np.zeros((n, num_vars_per_job * m))

        return flattened_allocation

    return unflattened_allocation


def print_effective_throughputs_v2(allocation, throughputs,
                                   job_id_to_job_type):
    """Prints effective throughputs given a job-job_type allocation."""
    job_ids = sorted(allocation.keys())
    worker_types = sorted(allocation[job_ids[0]].keys())
    job_types = sorted(set([job_id_to_job_type[job_id] for job_id in job_ids]))

    normalizing_factors = {}
    for job_id in job_ids:
        normalizing_factors[job_id] = 0.0
        job_type = job_id_to_job_type[job_id]
        for worker_type in worker_types:
            normalizing_factors[job_id] += \
                throughputs[worker_type][job_type]['null']

    effective_throughputs = {}
    for job_id in job_ids:
        job_type = job_id_to_job_type[job_id]
        effective_throughput = 0.0
        for worker_type in worker_types:
            throughput = \
                throughputs[worker_type][job_type]['null']
            throughput /= normalizing_factors[job_id]
            effective_throughput += \
                allocation[job_id][worker_type][None] * throughput
            for colocated_job_type in job_types:
                throughput = \
                    throughputs[worker_type][job_type][colocated_job_type][0]
                throughput /= normalizing_factors[job_id]
                effective_throughput += \
                    (allocation[job_id][worker_type][colocated_job_type] *\
                        throughput)
        print('%s: %f' % (job_id, effective_throughput))


def print_effective_throughputs(allocation, throughputs, job_id_to_job_type):
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
        job_type = job_id_to_job_type[single_job_id]
        for worker_type in worker_types:
            normalizing_factors[single_job_id] += \
                throughputs[worker_type][job_type]['null']

    effective_throughputs = {}
    for single_job_id in single_job_ids:
        job_type = job_id_to_job_type[single_job_id]
        effective_throughput = 0.0
        for job_id in relevant_job_ids[single_job_id]:
            for worker_type in worker_types:
                if not job_id.is_pair():
                    throughput = throughputs[worker_type][job_type]['null']
                else:
                    index = job_id.singletons().index(single_job_id)
                    other_job_id = \
                        job_id.singletons()[1 - index]
                    other_job_type = \
                        job_id_to_job_type[other_job_id]
                    throughput = \
                        throughputs[worker_type][job_type][other_job_type][0]
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
    policy = utils.get_policy('max_min_fairness_packed', solver=args.solver)
    scale_factors = {
        job.job_id: job.scale_factor for job in jobs
    }
    priority_weights = {
        job.job_id: job.priority_weight for job in jobs
    }

    start = datetime.datetime.now()
    v1_allocation = get_allocation_v1(policy, jobs, oracle_throughputs,
                                      cluster_spec, worker_types,
                                      scale_factors, priority_weights,
                                      flatten=False)
    v1_runtime = datetime.datetime.now() - start

    start = datetime.datetime.now()
    v3_allocation = get_allocation_v3(policy, jobs, oracle_throughputs,
                                      cluster_spec, worker_types,
                                      scale_factors, priority_weights,
                                      flatten=False)
    job_id_to_job_type = {job.job_id: job.job_type for job in jobs}
    v3_allocation = \
        convert_job_type_allocation_v2(v3_allocation, job_id_to_job_type)
    v3_runtime = datetime.datetime.now() - start

    if args.verbose:
        print('v1 allocation:')
        utils.print_allocation(v1_allocation)
        print('')

        print('v3 allocation:')
        utils.print_allocation(v3_allocation)
        print('')

        print('v1 effective_throughputs:')
        print_effective_throughputs(v1_allocation, oracle_throughputs,
                                    job_id_to_job_type)
        print('')

        print('v3 effective_throughputs:')
        print_effective_throughputs(v3_allocation, oracle_throughputs,
                                    job_id_to_job_type)

    print('v1 runtime:', v1_runtime.seconds + v1_runtime.microseconds / 1.0e6)
    print('v3 runtime:', v3_runtime.seconds + v3_runtime.microseconds / 1.0e6)

if __name__=='__main__':
    #test_v2()
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
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose')
    args = parser.parse_args()
    main(args)
