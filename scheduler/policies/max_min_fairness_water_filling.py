import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy


class MaxMinFairnessWaterFillingPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MaxMinFairnessWaterFilling_Perf'
        self._proportional_policy = ProportionalPolicy()

    def _get_allocation_helper(self, throughputs, priority_weights,
                               scale_factors_array, m, n, x_so_far):
        x = cp.Variable(throughputs.shape)
        c = cp.Variable()
        objective = cp.Maximize(c)

        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        scaled_effective_throughputs = cp.sum(cp.multiply(
            np.multiply(throughputs * priority_weights.reshape((m, 1)),
                        scale_factors_array), x), axis=1)

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array, x_so_far)
        constraints += [scaled_effective_throughputs == c]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return x

    def get_allocation(self, original_unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, original_cluster_spec):
        unflattened_throughputs = copy.deepcopy(original_unflattened_throughputs)        
        cluster_spec = copy.deepcopy(original_cluster_spec)

        done = False
        original_throughputs, original_index = super().flatten(original_unflattened_throughputs,
                                                               cluster_spec)
        (original_job_ids, original_worker_types) = original_index
        x_final = np.zeros(original_throughputs.shape)
        num_iterations = 0
        while not done:
            throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec)
            if throughputs is None: return None
            (m, n) = throughputs.shape
            (job_ids, worker_types) = index

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

            x_so_far = np.zeros((m, n))
            for i, job_id in enumerate(job_ids):
                for j, worker_type in enumerate(worker_types):
                    original_i = original_job_ids.index(job_id)
                    original_j = original_worker_types.index(worker_type)
                    x_so_far[i][j] = x_final[original_i][original_j]
            x = self._get_allocation_helper(
                throughputs, priority_weights, scale_factors_array, m, n, x_so_far)
            for i, job_id in enumerate(job_ids):
                for j, worker_type in enumerate(worker_types):
                    original_i = original_job_ids.index(job_id)
                    original_j = original_worker_types.index(worker_type)
                    x_final[original_i][original_j] += x.value[i][j]
            num_resources_used = np.multiply(x.value, scale_factors_array).sum(axis=0)

            worker_types_ordered = ['v100', 'p100', 'k80']
            worker_type_indices = {
                worker_types_ordered[i]: [worker_types.index(worker_type)
                                          for worker_type in worker_types_ordered[:i+1]]
                for i in range(len(worker_types_ordered))}
            print(x.value, x_final)
            job_id_set = set()
            for worker_type in worker_types_ordered:
                i = worker_types.index(worker_type)
                if cluster_spec[worker_type] > 0:
                    if num_resources_used[i] / cluster_spec[worker_type] < 0.999:
                        print("Resource %s underused (fraction used = %.2f)!" % (
                            worker_type, num_resources_used[i] / cluster_spec[worker_type]))
                        for j, job_id in enumerate(job_ids):
                            if np.sum(x_final[original_job_ids.index(job_id)][worker_type_indices[worker_type]]) >= 0.999:
                                if job_id not in job_id_set:
                                    print("Deleting job %s because of worker_type %s" % (job_id, worker_type))
                                    job_id_set.add(job_id)
                    cluster_spec[worker_type] -= num_resources_used[i]
                    if cluster_spec[worker_type] < 1e-1:
                        cluster_spec[worker_type] = 0
            if len(job_id_set) == 0:
                done = True
            for job_id in job_id_set:
                del unflattened_throughputs[job_id]
            print("Cluster spec:", cluster_spec)
            print()
            print()
            num_iterations += 1
            if len(unflattened_throughputs) == 0 or np.all([cluster_spec[worker_type] == 0 for worker_type in worker_types]):
                done = True
        print("Number of iterations: %d" % num_iterations)

        return super().unflatten(x_final.clip(min=0.0).clip(max=1.0), index)
