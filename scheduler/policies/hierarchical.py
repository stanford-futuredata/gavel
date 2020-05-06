import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy
from max_min_fairness import MaxMinFairnessPolicyWithPerf
from max_min_fairness_multi_entity import MaxMinFairnessMultiEntityPolicyWithPerf

class TwoLevelSinglePassHierarchicalPolicy(Policy):

    def __init__(self, solver):
        self._name = 'TwoLevelSinglePassHierarchical'
        self._top_level_policy = MaxMinFairnessMultiEntityPolicyWithPerf(solver)
        self._bottom_level_policy = MaxMinFairnessPolicyWithPerf(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, job_to_entity_mapping,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        top_level_allocation = self._top_level_policy.get_allocation(
            unflattened_throughputs, scale_factors, unflattened_priority_weights,
            job_to_entity_mapping, cluster_spec)

        entity_ids = list(unflattened_priority_weights.keys())
        entity_to_job_mapping = {}
        for job_id in job_ids:
            entity_id = job_to_entity_mapping[job_id]
            if entity_id not in entity_to_job_mapping:
                entity_to_job_mapping[entity_id] = []
            entity_to_job_mapping[entity_id].append(job_id)

        bottom_level_allocation = {}
        for entity_id in entity_ids:
            new_cluster_spec = {}
            for worker_type in cluster_spec:
                new_cluster_spec[worker_type] = 0.0
                for job_id in entity_to_job_mapping[entity_id]:
                    new_cluster_spec[worker_type] += \
                        top_level_allocation[job_id][worker_type]
            unflattened_throughputs_filtered = {
                job_id: unflattened_throughputs[job_id]
                for job_id in entity_to_job_mapping[entity_id]
            }
            priority_weights = {
                job_id: 1.0 for job_id in entity_to_job_mapping[entity_id]}
            allocation = self._bottom_level_policy.get_allocation(
                unflattened_throughputs_filtered, scale_factors, priority_weights,
                new_cluster_spec)
            for job_id in allocation:
                bottom_level_allocation[job_id] = allocation[job_id]

        return bottom_level_allocation