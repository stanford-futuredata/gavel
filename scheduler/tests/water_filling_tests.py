import sys; sys.path.append("..")
from policies import max_min_fairness_water_filling

import random

def test_water_filling():
    policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
        solver='ECOS')
    worker_types = ['v100', 'p100', 'k80']
    cluster_spec = {worker_type: 1 for worker_type in worker_types}
    num_jobs = 3
    print("Total number of jobs: %d" % num_jobs)
    unflattened_throughputs = {}
    scale_factors = {}
    unflattened_priority_weights = {}
    num_workers_requested = 0
    for i in range(num_jobs):
        throughputs = [random.random() for i in range(len(worker_types))]
        throughputs.sort(reverse=True)
        unflattened_throughputs[i] = {
            worker_types[i]: throughputs[i] for i in range(len(worker_types))}
        scale_factors[i] = 1
        num_workers_requested += scale_factors[i]
        unflattened_priority_weights[i] = random.randint(1, 5)
        print(i, unflattened_throughputs[i], unflattened_priority_weights[i])
    print("Total number of workers requested: %d" % num_workers_requested)
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       cluster_spec)
    print()


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    for i in range(5):
        test_water_filling()
