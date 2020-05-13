import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy


class MaxMinFairnessWaterFillingPolicyWithPerf(Policy):

    def __init__(self):
        Policy.__init__(self, solver=None)
        self._name = 'MaxMinFairnessWaterFilling_Perf'
        self._proportional_policy = ProportionalPolicy()

    def recompute_priority_weights(self):
        pass

    def _get_allocation_helper(self, throughputs, index, priority_weights,
                               scale_factors_array, m, n,
                               per_job_max_c,
                               computed_c=None):
        x = cp.Variable(throughputs.shape)
        c = cp.Variable()
        (job_ids, _) = index
        M = np.max(np.multiply(throughputs * priority_weights.reshape((m, 1)),
                               scale_factors_array))
        epsilon = 1e-5

        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        scaled_effective_throughputs = cp.sum(cp.multiply(
            np.multiply(throughputs * priority_weights.reshape((m, 1)),
                        scale_factors_array), x), axis=1)

        if computed_c is None:
            objective = cp.Maximize(c)
        else:
            z = cp.Variable(m, boolean=True)
            objective = cp.Maximize(cp.sum(z))

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        for i, job_id in enumerate(job_ids):
            if job_id in per_job_max_c:
                constraints.append(
                    scaled_effective_throughputs[i] == per_job_max_c[job_id])
                if computed_c is not None:
                    constraints.append(z[i] == 0)
            else:
                if computed_c is not None:
                    constraints.append(
                        scaled_effective_throughputs[i] >= computed_c)
                    constraints.append(
                        (M * z[i]) >=
                            (scaled_effective_throughputs[i] - (computed_c * 1.0001) + epsilon))
                    constraints.append(
                        (M * (1 - z[i])) >=
                            ((computed_c * 1.0001) - scaled_effective_throughputs[i]))
                else:
                    constraints.append(
                        scaled_effective_throughputs[i] == c)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver='ECOS' if computed_c is None else 'GLPK_MI')

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        if computed_c is None:
            return x, objective.value
        return x, z.value

    def get_allocation(self, original_unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, original_cluster_spec,
                       verbose=False):
        unflattened_throughputs = copy.deepcopy(original_unflattened_throughputs)        
        cluster_spec = copy.deepcopy(original_cluster_spec)

        done = False
        original_throughputs, original_index = super().flatten(original_unflattened_throughputs,
                                                               cluster_spec)
        (original_job_ids, original_worker_types) = original_index
        per_job_max_c = {}
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

            x, c = self._get_allocation_helper(
                throughputs, index, priority_weights, scale_factors_array,
                m, n, per_job_max_c=per_job_max_c)
            if num_iterations == 0:
                print("Objective value: %.3f" % c)

            # Find bottleneck job_ids.
            _, z = self._get_allocation_helper(
                throughputs, index, priority_weights, scale_factors_array, m, n,
                per_job_max_c=per_job_max_c, computed_c=c)
            for i, job_id in enumerate(job_ids):
                if job_id not in per_job_max_c and (z is None or not z[i]):
                    print("Iteration %d:" % num_iterations, job_id)
                    per_job_max_c[job_id] = c
            if verbose:
                print("At the end of iteration %d:" % num_iterations,
                    x.value)
            num_iterations += 1
            self.recompute_priority_weights()
            if len(unflattened_throughputs) == len(per_job_max_c):
                done = True
        print("Number of iterations: %d" % num_iterations)

        (m, n) = len(original_job_ids), len(original_worker_types)
        scale_factors_array = self.scale_factors_array(
            scale_factors, original_job_ids, m, n)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in original_job_ids])

        proportional_throughputs = self._proportional_policy.get_throughputs(
            original_throughputs, original_index, original_cluster_spec)
        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))
        scaled_effective_throughputs = np.sum(np.multiply(
            np.multiply(original_throughputs * priority_weights.reshape((m, 1)),
                        scale_factors_array), x.value), axis=1)

        print("Scaled effective throughputs:", scaled_effective_throughputs)
        print("Final objective: %.3f" % np.min(scaled_effective_throughputs))
        print("Constraints:",
            np.multiply(x.value, scale_factors_array).sum(axis=0),
            x.value.sum(axis=1))

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
