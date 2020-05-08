import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy
from proportional import ProportionalPolicy

class MaxMinFairnessMultiEntityPolicy(Policy):

    def __init__(self, solver):
        self._name = 'MaxMinFairnessMultiEntity'
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessMultiEntityPolicyWithPerf(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       priority_weights, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = \
                     unflattened_throughputs[job_id]['v100']

        return self._max_min_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, priority_weights,
            cluster_spec)


class MaxMinFairnessMultiEntityPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MaxMinFairnessMultiEntity_Perf'
        self._proportional_policy = ProportionalPolicy()

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, job_to_entity_mapping,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        entity_ids = list(unflattened_priority_weights.keys())
        entity_to_job_mapping = {}
        for job_id in job_ids:
            entity_id = job_to_entity_mapping[job_id]
            if entity_id not in entity_to_job_mapping:
                entity_to_job_mapping[entity_id] = []
            entity_to_job_mapping[entity_id].append(job_id)
        entity_masks = []
        for entity_id in entity_ids:
            entity_mask = np.zeros(m)
            for i, job_id in enumerate(job_ids):
                if job_id in entity_to_job_mapping[entity_id]:
                    entity_mask[i] = 1.0
            entity_masks.append(entity_mask)

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        priority_weights = [1. / unflattened_priority_weights[entity_id]
                            for entity_id in entity_ids]

        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)
        scaling_factors = 1.0 / proportional_throughputs.reshape((m, 1))

        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        scaled_effective_throughputs = cp.sum(cp.multiply(
            np.multiply(throughputs * scaling_factors.reshape((m, 1)),
                        scale_factors_array), x), axis=1)
        objective = cp.Maximize(cp.minimum(*[
            priority_weights[i] * cp.sum(cp.multiply(
                entity_masks[i],
                scaled_effective_throughputs)) for i in range(len(entity_ids))]))
        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
