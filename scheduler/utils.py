import json
import os

import job
import policies

def get_available_policies():
    return ['fifo', 'fifo_perf', 'fifo_packed',
            'max_min_fairness',
            'max_min_fairness_perf',
            'max_min_fairness_packed',
            'min_total_duration',
            'min_total_duration_packed',
            'max_sum_throughput_perf',
            'max_sum_throughput_packed']

def read_all_throughputs_json(throughputs_file):
    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)
    return throughputs

def get_policy(policy_name, solver, seed=None):
    if policy_name == 'max_min_fairness':
        policy = policies.MaxMinFairnessPolicy(solver=solver)
    elif policy_name == 'max_min_fairness_perf':
        policy = policies.MaxMinFairnessPolicyWithPerf(solver=solver)
    elif policy_name == 'max_min_fairness_packed':
        policy = policies.MaxMinFairnessPolicyWithPacking(solver=solver)
    elif policy_name == 'max_sum_throughput_perf':
        policy = policies.MaxSumThroughputPolicyWithPerf(solver=solver)
    elif policy_name == 'max_sum_throughput_packed':
        policy = policies.MaxSumThroughputPolicyWithPacking(solver=solver)
    elif policy_name == 'min_total_duration':
        policy = policies.MinTotalDurationPolicy(solver=solver)
    elif policy_name == 'min_total_duration_packed':
        policy = policies.MinTotalDurationPolicyWithPacking(solver=solver)
    elif policy_name == 'fifo':
        policy = policies.FIFOPolicy(seed=seed)
    elif policy_name == 'fifo_perf':
        policy = policies.FIFOPolicyWithPerf()
    elif policy_name == 'fifo_packed':
        policy = policies.FIFOPolicyWithPacking()
    else:
        raise ValueError('Unknown policy!')
    return policy

def parse_trace(trace_file, run_dir):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            (job_type, command, num_steps_arg, needs_data_dir, total_steps,
             arrival_time, scale_factor) = line.split('\t')
            if int(scale_factor) == 0:
                continue
            if int(needs_data_dir):
                command = command % (run_dir, run_dir)
            else:
                command = command % (run_dir)
            jobs.append(job.Job(job_id=None,
                                job_type=job_type,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=int(total_steps),
                                duration=None,
                                scale_factor=int(scale_factor),
                                priority_weight=1.0))
            arrival_times.append(float(arrival_time))
    return jobs, arrival_times

def print_allocation(allocation, current_time=None):
    """Prints the allocation.

       Debug method used for printing the allocation of each job on each
       worker type.
    """
    print('=' * 80)
    if current_time is not None:
        print('Allocation\t(Current_time: %f)' % (current_time))
        print('-' * 80)
    for job_id in sorted(list(allocation.keys())):
        allocation_str = 'Job ID %s:' % (job_id)
        for worker_type in sorted(list(allocation[job_id].keys())):
            value = allocation[job_id][worker_type]
            allocation_str += ' [%s: %f]' % (worker_type, value)
        print(allocation_str)
    print('=' * 80)
