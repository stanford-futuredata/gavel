import sys; sys.path.append("..")
from policies import max_min_fairness_water_filling

import random
import time

def test_water_filling():
    policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
        priority_reweighting_policy=None)
    worker_types = ['v100', 'p100', 'k80']
    cluster_spec = {worker_type: 64 for worker_type in worker_types}
    num_jobs = 300
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
        scale_factors[i] = 2 ** random.randint(0, 2)
        num_workers_requested += scale_factors[i]
        unflattened_priority_weights[i] = random.randint(1, 5)
        print("Job %d: Throughputs: %s, Priority: %d, Scale factor: %d" % (
            i, unflattened_throughputs[i], unflattened_priority_weights[i],
            scale_factors[i]))
    print("Total number of workers requested: %d" % num_workers_requested)
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       cluster_spec)
    print()


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    start_time = time.time()
    for i in range(5):
        test_water_filling()
    print("Average time per problem: %.2f seconds" % ((time.time() - start_time) / 5))
