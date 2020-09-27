import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy

class MaxMinFairnessStrategyProofPolicy(Policy):

    def __init__(self, solver):
        self._name = 'MaxMinFairness'
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessPolicyWithPerf(solver)

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
                 new_unflattened_throughputs[job_id][worker_type] = 1.0

        return self._max_min_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, priority_weights,
            cluster_spec)


class MaxMinFairnessStrategyProofPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MaxMinFairness_Perf'
        self._proportional_policy = ProportionalPolicy()

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec, recurse_deeper=True):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        if recurse_deeper:
            all_throughputs_minus_job = []
            for job_id in job_ids:
                unflattened_throughputs_minus_job = copy.copy(unflattened_throughputs)
                del unflattened_throughputs_minus_job[job_id]
                throughputs_minus_job = self.get_allocation(
                    unflattened_throughputs_minus_job, scale_factors,
                    unflattened_priority_weights, cluster_spec,
                    recurse_deeper=False)
                all_throughputs_minus_job.append(throughputs_minus_job)

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)
        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))

        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.geo_mean(cp.sum(cp.multiply(
                np.multiply(throughputs * priority_weights.reshape((m, 1)),
                            scale_factors_array), x), axis=1)))
        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        throughputs = np.sum(np.multiply(throughputs, x.value), axis=1)
        throughputs_dict = {job_ids[i]: throughputs[i] for i in range(len(job_ids))}
        if not recurse_deeper:
            return throughputs_dict

        discount_factors = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            discount_factor = 1.0
            for other_job_id in all_throughputs_minus_job[i]:
                discount_factor *= (
                    throughputs_dict[other_job_id] / all_throughputs_minus_job[i][other_job_id])
            discount_factors[i] = discount_factor

        discounted_allocation = np.multiply(x.value.T, discount_factors).T

        return super().unflatten(discounted_allocation.clip(min=0.0).clip(max=1.0), index), \
            discount_factors
