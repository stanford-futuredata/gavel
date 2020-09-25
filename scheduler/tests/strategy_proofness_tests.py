import sys; sys.path.append("..")
from policies import max_min_fairness, max_min_fairness_strategy_proof

import random
import time

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def test_strategy_proofness(unflattened_throughputs, cluster_spec):
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
        print("Job %d: Throughputs=%s, Priority=%d, Scale factor=%d" % (
            i, unflattened_throughputs[i], unflattened_priority_weights[i],
            scale_factors[i]))
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


if __name__ == '__main__':
    worker_types = ['v100', 'p100', 'k80']
    # throughputs specified in the above order.
    honest_throughputs = [
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0]
    ]
    dishonest_throughputs = [
        [2.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0]
    ]
    cluster_spec = {
        'v100': 1, 'p100': 1, 'k80': 1
    }
    for (throughputs, experiment_string) in zip(
        [honest_throughputs, dishonest_throughputs],
        ["Allocation computed using true throughputs",
         "Allocation computed using dishonest throughputs"]):
        print("=" * 100)
        print(experiment_string)
        print("=" * 100)

        throughputs = np.array(throughputs)
        unflattened_throughputs = {
            i: {worker_types[j]: throughputs[i][j] for j in range(len(worker_types))}
            for i in range(len(throughputs))
        }
        unflattened_allocation, discount_factors, unflattened_non_strategy_proof_allocation, runtime = \
            test_strategy_proofness(unflattened_throughputs, cluster_spec)
        allocation = np.zeros((len(throughputs), len(cluster_spec)))
        non_strategy_proof_allocation = np.zeros((len(throughputs), len(cluster_spec)))
        for i in range(len(throughputs)):
            for j in range(len(worker_types)):
                allocation[i][j] = unflattened_allocation[i][worker_types[j]]
                non_strategy_proof_allocation[i][j] = \
                    unflattened_non_strategy_proof_allocation[i][worker_types[j]]
        effective_throughputs = np.sum(np.multiply(throughputs, allocation),
                                       axis=1)
        print("Strategy-proof allocation:", allocation)
        print("Discount factors:", discount_factors)
        effective_throughputs = np.sum(np.multiply(throughputs, allocation),
                                       axis=1)
        print("Effective_throughputs with advertised raw throughputs:", effective_throughputs)
        effective_throughputs = np.sum(np.multiply(np.array(honest_throughputs), allocation),
                                       axis=1)
        print("Effective_throughputs with true raw throughputs:", effective_throughputs)
        print("-" * 100)
        print("Standard max-min fairness allocation", non_strategy_proof_allocation)
        effective_throughputs = np.sum(np.multiply(throughputs, non_strategy_proof_allocation),
                                       axis=1)
        print("Effective_throughputs with advertised raw throughputs:", effective_throughputs)
        effective_throughputs = np.sum(np.multiply(np.array(honest_throughputs), non_strategy_proof_allocation),
                                       axis=1)
        print("Effective_throughputs with true raw throughputs:", effective_throughputs)

    print("=" * 100)
    print("Average time: %.2f seconds" % runtime)
    print()
