import sys; sys.path.append("..")
from policies import max_min_fairness_water_filling

import random
import time

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def test_water_filling_multilevel(priority_reweighting_policy):
    policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
        priority_reweighting_policy=priority_reweighting_policy)
    worker_types = ['k80', 'p100', 'v100']
    cluster_spec = {worker_type: 2 for worker_type in worker_types}
    num_jobs = 10
    num_entities = 2
    print("Total number of jobs: %d" % num_jobs)
    print("Total number of entities: %d" % num_entities)
    unflattened_throughputs = {}
    scale_factors = {}
    unflattened_priority_weights = {}
    entity_to_job_mapping = {}
    num_workers_requested = 0
    for i in range(num_entities):
        entity_id = 'entity%d' % i
        entity_to_job_mapping[entity_id] = []
        # unflattened_priority_weights[entity_id] = random.randint(1, 5)
        unflattened_priority_weights[entity_id] = random.randint(1, 2)
        print("Entity %d: Priority: %d" % (
            i, unflattened_priority_weights[entity_id]))
    for i in range(num_jobs):
        # throughputs = [random.random() for i in range(len(worker_types))]
        throughputs = [3.0, 2.0, 1.0]
        throughputs.sort()
        unflattened_throughputs[i] = {
            worker_types[i]: throughputs[i] for i in range(len(worker_types))}
        # scale_factors[i] = 2 ** random.randint(0, 2)
        scale_factors[i] = 1
        num_workers_requested += scale_factors[i]
        priority_weight = random.randint(1, 2)
        if priority_reweighting_policy == 'fairness+fifo':
            priority_weight = 1.0
        unflattened_priority_weights[i] = priority_weight
        # entity_id = 'entity%d' % random.randint(0, num_entities-1)
        entity_id = 'entity%d' % (i // 5)
        entity_to_job_mapping[entity_id].append(i)
        print("Job %d: Throughputs: %s, Priority: %d, Scale factor: %d, Entity: %s" % (
            i, unflattened_throughputs[i], unflattened_priority_weights[i],
            scale_factors[i], entity_id.replace('entity', '')))
    print("Total number of workers requested: %d" % num_workers_requested)
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       cluster_spec,
                                       entity_to_job_mapping=entity_to_job_mapping,
                                       verbose=True)
    print()


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    start_time = time.time()
    priority_reweighting_policy = 'fairness+fifo'
    for i in range(5):
        test_water_filling_multilevel(priority_reweighting_policy)
    print("Average time per problem: %.2f seconds" % ((time.time() - start_time) / 5))
