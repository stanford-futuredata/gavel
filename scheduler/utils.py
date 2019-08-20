import json
import os

import policies

def read_all_throughputs_json(throughputs_file):
    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)
    return throughputs

def get_policy(policy_name, seed=None):
    if policy_name == 'least_attained_service':
        policy = policies.LeastAttainedServicePolicy()
    elif policy_name == 'least_attained_service_perf':
        policy = policies.LeastAttainedServicePolicyWithPerf()
    elif policy_name == 'least_attained_service_packed':
        policy = policies.LeastAttainedServicePolicyWithPacking()
    elif policy_name == 'max_min_fairness':
        policy = policies.MaxMinFairnessPolicy()
    elif policy_name == 'max_min_fairness_perf':
        policy = policies.MaxMinFairnessPolicyWithPerf()
    elif policy_name == 'max_min_fairness_packed':
        policy = policies.MaxMinFairnessPolicyWithPacking()
    elif policy_name == 'min_total_duration':
        policy = policies.MinTotalDurationPolicy()
    elif policy_name == 'min_total_duration_packed':
        policy = policies.MinTotalDurationPolicyWithPacking()
    elif policy_name == 'fifo':
        policy = policies.FIFOPolicy(seed=seed)
    elif policy_name == 'fifo_perf':
        policy = policies.FIFOPolicyWithPerf()
    elif policy_name == 'fifo_packed':
        policy = policies.FIFOPolicyWithPacking()
    else:
        raise ValueError('Unknown policy!')
    return policy
