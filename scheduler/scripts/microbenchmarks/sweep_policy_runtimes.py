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
                  run_dir='/tmp', job_template=None):
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
                    solver, oracle_throughputs, generate_multi_gpu_jobs,
                    generate_multi_priority_jobs, trial):
    policy = utils.get_policy(policy_name, solver=solver)
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
        jobs[job_id] = \
            _generate_job(rng, oracle_throughputs,
                          generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                          generate_multi_priority_jobs=generate_multi_priority_jobs)
        throughputs[job_id] = {}
        for worker_type in cluster_spec:
            throughputs[job_id][worker_type] = \
                oracle_throughputs[worker_type][jobs[job_id].job_type]['null']
    if '_Packing' in policy.name:
        for i in range(num_active_jobs):
            job_id_0 = JobIdPair(i, None)
            for j in range(num_active_jobs):
                job_id_1 = JobIdPair(j, None)
                if (i < j and jobs[job_id_0].scale_factor ==
                    jobs[job_id_1].scale_factor):
                    throughputs[JobIdPair(i, j)] = {}
                    for worker_type in cluster_spec:
                        throughputs[JobIdPair(i, j)][worker_type] = \
                            oracle_throughputs[worker_type][jobs[job_id_0].job_type][jobs[job_id_1].job_type]
    scale_factors = {
        job_id: jobs[job_id].scale_factor for job_id in jobs
    }
    start_time = time.time()
    if policy.name.startswith('MaxMinFairness'):
        if policy.name == 'MaxMinFairness_Packing':
            seen_job_types = {} # Map from job type to job ID.
            app_throughputs = {}
            app_scale_factors = {}
            job_id_to_app_job_id = {}
            setup_start_time = time.time()
            sorted_job_ids = sorted(throughputs.keys())
            for i, job_id in enumerate(sorted_job_ids):
                if not job_id.is_pair():
                    if not (jobs[job_id].job_type, None) in seen_job_types:
                        seen_job_types[(jobs[job_id].job_type, None)] = job_id
                        app_throughputs[job_id] = throughputs[job_id]
                        app_scale_factors[job_id] = scale_factors[job_id]
                        job_id_to_app_job_id[job_id] = job_id
                    else:
                        seen_job_id = \
                            seen_job_types[(jobs[job_id].job_type, None)]
                        app_scale_factors[seen_job_id] += scale_factors[job_id]
                        job_id_to_app_job_id[job_id] = seen_job_id
                else:
                    single_job_ids = job_id.singletons()
                    if (scale_factors[single_job_ids[0]] !=
                        scale_factors[single_job_ids[1]]):
                        continue
                    job_types = []
                    for single_job_id in single_job_ids:
                        job_types.append(jobs[single_job_id].job_type)
                    if ((job_types[0], job_types[1]) not in seen_job_types and
                        (job_types[1], job_types[0]) not in seen_job_types):
                        seen_job_types[(job_types[0], job_types[1])] = job_id
                        app_throughputs[job_id] = throughputs[job_id]
                        app_scale_factors[job_id] = \
                            scale_factors[single_job_ids[0]]
                        job_id_to_app_job_id[job_id] = job_id
                    else:
                        try:
                            seen_job_id = \
                                seen_job_types[(job_types[0], job_types[1])]
                        except KeyError as e:
                            seen_job_id = \
                                seen_job_types[(job_types[1], job_types[0])]
                        app_scale_factors[seen_job_id] += \
                            scale_factors[single_job_ids[0]]
                        job_id_to_app_job_id[job_id] = seen_job_id
            setup_time = time.time() - setup_start_time
            app_allocation = policy.get_allocation(app_throughputs,
                                                   app_scale_factors, None,
                                                   cluster_spec)
            allocation = {}
            for job_id in sorted_job_ids:
                allocation[job_id] = \
                    app_allocation[job_id_to_app_job_id[job_id]]

        # TODO: Fix priority weights also
        priority_weights = {
            job_id: jobs[job_id].priority_weight for job_id in jobs
        }
        scale_factors = {
            job_id: jobs[job_id].scale_factor for job_id in jobs
        }
        original_allocation = policy.get_allocation(
            throughputs, scale_factors, priority_weights,
            cluster_spec)
        deltas = []
        for job_id in sorted_job_ids:
            print('Job id %s' % (job_id))
            print('Optimized allocation:', allocation[job_id])
            print('Original allocation:', original_allocation[job_id])
            print('')
            for worker_type in allocation[job_id]:
                deltas.append(allocation[job_id][worker_type] -
                              original_allocation[job_id][worker_type])
        print('L2 norm:',
              np.sqrt(np.sum([np.square(delta) for delta in deltas])))

    elif policy.name.startswith('MinTotalDuration'):
        num_steps_remaining = {
            job_id: jobs[job_id].num_steps for job_id in jobs
        }
        policy.get_allocation(
            throughputs, scale_factors, num_steps_remaining,
            cluster_spec)
    else:
        policy.get_allocation(
            throughputs, scale_factors, cluster_spec)

    runtime = time.time() - start_time
    print('%s,%d,%s,%s,%d,%f' % (cluster_spec_str, num_active_jobs,
                                 policy.name, solver, trial, runtime),
          flush=True)
    print('Setup time = %f' % (setup_time), flush=True)

def main(args):
    all_cluster_specs = args.cluster_spec
    all_num_active_jobs = args.num_active_jobs
    all_policies = args.policies
    all_solvers = args.solvers
    num_trials = args.num_trials
    oracle_throughputs = utils.read_all_throughputs_json(args.throughputs_file)

    print('Cluster spec,# Active Jobs,Policy,Solver,Trial,Runtime')
    for cluster_spec in all_cluster_specs:
        for num_active_jobs in all_num_active_jobs:
            for policy in all_policies:
                for solver in all_solvers:
                    for trial in range(num_trials):
                        measure_runtime(cluster_spec, num_active_jobs, policy,
                                        solver, oracle_throughputs,
                                        args.generate_multi_gpu_jobs,
                                        args.generate_multi_priority_jobs,
                                        trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--throughputs-file', type=str,
                        default=('/lfs/1/keshav2/gpusched/scheduler/'
                                 'oracle_throughputs.json'),
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
    parser.add_argument('--solvers', type=str, nargs='+', default='ECOS',
                        help='List of solvers to sweep through')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials to run for each experiment')
    args = parser.parse_args()

    main(args)
