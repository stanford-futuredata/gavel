import sys; sys.path.append("..")
from policies import max_min_fairness, max_min_fairness_strategy_proof

import random
import time

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def test_strategy_proofness(unflattened_throughputs, cluster_spec, verbose=True):
    policy = max_min_fairness_strategy_proof.MaxMinFairnessStrategyProofPolicyWithPerf(
        solver='ECOS')
    non_strategy_proof_policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(
        solver='ECOS')
    num_jobs = len(unflattened_throughputs)
    worker_types = list(cluster_spec.keys())
    scale_factors = {}
    unflattened_priority_weights = {}
    num_workers_requested = 0
    for i in range(num_jobs):
        scale_factors[i] = 1
        num_workers_requested += scale_factors[i]
        unflattened_priority_weights[i] = 1
        if verbose:
            print("Job %d: Throughputs=%s, Priority=%d, Scale factor=%d" % (
                i, unflattened_throughputs[i], unflattened_priority_weights[i],
                scale_factors[i]))
    if verbose:
        print("-" * 100)
    start_time = time.time()
    allocation, discount_factors = policy.get_allocation(
        unflattened_throughputs, scale_factors,
        unflattened_priority_weights,
        cluster_spec)
    non_strategy_proof_allocation = non_strategy_proof_policy.get_allocation(
        unflattened_throughputs, scale_factors,
        unflattened_priority_weights,
        cluster_spec)
    return allocation, discount_factors, non_strategy_proof_allocation, time.time() - start_time
