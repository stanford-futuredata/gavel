import sys; sys.path.append("..")
from policies import max_min_fairness_water_filling

import random

def test_two_level_single_pass_hierarchical():
    policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
        solver='ECOS')
    worker_types = ['v100', 'p100', 'k80']
    cluster_spec = {worker_type: 32 for worker_type in worker_types}
    num_jobs = random.randint(1, 100)
    unflattened_throughputs = {}
    scale_factors = {}
    unflattened_priority_weights = {}
    for i in range(num_jobs):
        throughputs = [random.random() for i in range(len(worker_types))]
        throughputs.sort(reverse=True)
        unflattened_throughputs[i] = {
            worker_types[i]: throughputs[i] for i in range(len(worker_types))}
        scale_factors[i] = 2 ** random.choice(range(4))
        unflattened_priority_weights[i] = random.randint(1, 5)
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       cluster_spec)
    return allocation


if __name__ == '__main__':
    allocation = test_two_level_single_pass_hierarchical()
