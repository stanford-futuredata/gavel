import sys; sys.path.append("..")
from policies import max_min_fairness_water_filling

import random
import time

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def test_water_filling_multilevel():
    num_entities = 10
    priority_reweighting_policies = {}
    unflattened_priority_weights = {}
    entity_to_job_mapping = {}
    entity_weights = {}
    for i in range(num_entities):
        entity_id = 'entity%d' % i
        priority_reweighting_policies[entity_id] = [
            'fifo', 'fairness'][random.randint(0, 1)]
        entity_to_job_mapping[entity_id] = []
        entity_weights[entity_id] = random.randint(1, 5)
        print("Entity %d: Priority=%d, Policy=%s" % (
            i, entity_weights[entity_id],
            priority_reweighting_policies[entity_id]))

    policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
        priority_reweighting_policies=priority_reweighting_policies)
    worker_types = ['k80', 'p100', 'v100']
    cluster_spec = {worker_type: 64 for worker_type in worker_types}
    num_jobs = 300
    print("Total number of jobs: %d" % num_jobs)
    print("Total number of entities: %d" % num_entities)
    unflattened_throughputs = {}
    scale_factors = {}
    num_workers_requested = 0
    for i in range(num_jobs):
        throughputs = [random.random() for i in range(len(worker_types))]
        throughputs.sort()
        unflattened_throughputs[i] = {
            worker_types[i]: throughputs[i] for i in range(len(worker_types))}
        scale_factors[i] = 2 ** random.randint(0, 2)
        num_workers_requested += scale_factors[i]
        entity_id = 'entity%d' % random.randint(0, num_entities-1)
        priority_weight = random.randint(1, 5)
        if priority_reweighting_policies[entity_id] == 'fifo':
            priority_weight = 1.0
        unflattened_priority_weights[i] = priority_weight
        entity_to_job_mapping[entity_id].append(i)
        print("Job %d: Throughputs=%s, Priority=%d, Scale factor=%d, Entity=%s" % (
            i, unflattened_throughputs[i], unflattened_priority_weights[i],
            scale_factors[i], entity_id.replace('entity', '')))
    print("Total number of workers requested: %d" % num_workers_requested)
    start_time = time.time()
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       cluster_spec,
                                       entity_weights=entity_weights,
                                       entity_to_job_mapping=entity_to_job_mapping,
                                       verbose=True)
    print()
    return time.time() - start_time


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    priority_reweighting_policy = 'fifo'
    times = []
    for i in range(5):
        times.append(test_water_filling_multilevel())
    print("Average time per problem: %.2f seconds" % np.mean(np.array(times)))
