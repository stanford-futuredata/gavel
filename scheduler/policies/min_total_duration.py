import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class MinTotalDurationPolicy(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MinTotalDuration_Perf'

    def get_allocation_helper(self, throughputs, scale_factors_array, T):
        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(1)
        # Make sure the allocation can fit in the cluster, and that the
        # currently active jobs can finish in time T.
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= self._num_workers,
            cp.sum(x, axis=1) <= 1,
            cp.sum(cp.multiply(throughputs, x), axis=1) >= (
                self._num_steps_remaining / T),
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        return cvxprob.status, x

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       num_steps_remaining, cluster_spec):
        # TODO: Might not want to normalize throughputs here.
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if index is None: return None
        (m, n) = throughputs.shape
        (job_ids, _) = index
        self._num_steps_remaining = np.array([num_steps_remaining[job_id]
                                              for job_id in job_ids])
        if throughputs is None: return None

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]

        # Units are in seconds.
        max_T = 1000000.
        min_T = 100.
        last_max_T = max_T
        status = None
        last_feasible_x = None
        while last_feasible_x is None:
            # Binary search for the smallest T that gives an optimal solution.
            while (1.05 * min_T) < max_T:  # TODO: Can tweak the 1.05 in this loop.
                T = (min_T + max_T) / 2.
                status, x = self.get_allocation_helper(throughputs,
                                                       scale_factors_array, T)
                if status == "optimal":
                    last_feasible_x = x
                    max_T = T
                else:
                    min_T = T
            max_T = last_max_T * 10.
            min_T = last_max_T
            last_max_T *= 10

        assert(last_feasible_x is not None)
        return super().unflatten(
            last_feasible_x.value.clip(min=0.0).clip(max=1.0), index)


class MinTotalDurationPolicyWithPacking(PolicyWithPacking):

    def __init__(self, solver):
        PolicyWithPacking.__init__(self, solver)
        self._name = 'MinTotalDuration_Packing'

    def get_allocation_helper(self, all_throughputs, job_ids,
                              single_job_ids, scale_factors_array, T,
                              relevant_combinations):
        x = cp.Variable(all_throughputs[0].shape)
        objective = cp.Maximize(1)
        # Make sure the allocation can fit in the cluster.
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= self._num_workers,
        ]

        # Every job cannot receive a total time share sum greater than 1.0.
        for single_job_id in single_job_ids:
            indexes = relevant_combinations[single_job_id]
            constraints.append(cp.sum(x[indexes]) <= 1)
        for i, (throughputs, num_steps_remaining) in \
            enumerate(zip(all_throughputs, self._num_steps_remaining)):
            indexes = relevant_combinations[single_job_ids[i]]
            # Ensure that every job satisfies its throughput constraint,
            # and can finish in time T.
            constraints.append(
                cp.sum(cp.multiply(throughputs[indexes], x[indexes])) >=
                    (num_steps_remaining / T))
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        return cvxprob.status, x

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       num_steps_remaining, cluster_spec):
        all_throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec, normalize=False)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        if index is None: return None
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        self._num_steps_remaining = [num_steps_remaining[single_job_id]
                                     for single_job_id in single_job_ids]

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        (m, n) = all_throughputs[0].shape
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            scale_factor = None
            for single_job_id in job_ids[i].singletons():
                if (scale_factor is not None and
                    scale_factor != scale_factors[single_job_id]):
                    scale_factor = 0
                else:
                    scale_factor = scale_factors[single_job_id]
            for j in range(n):
                scale_factors_array[i, j] = scale_factor

        # Units are in seconds.
        max_T = 1000000.
        min_T = 100.
        last_max_T = max_T
        status = None
        last_feasible_x = None
        while last_feasible_x is None:
            # Binary search for the smallest T that gives an optimal solution.
            while (1.05 * min_T) < max_T:  # TODO: Can tweak the 1.05 in this loop.
                T = (min_T + max_T) / 2.
                status, x = self.get_allocation_helper(all_throughputs,
                                                       job_ids, single_job_ids,
                                                       scale_factors_array, T,
                                                       relevant_combinations)
                if status == "optimal":
                    last_feasible_x = x
                    max_T = T
                else:
                    min_T = T
            max_T = last_max_T * 10.
            min_T = last_max_T
            last_max_T *= 10

        assert(last_feasible_x is not None)
        return super().unflatten(
            last_feasible_x.value.clip(min=0.0).clip(max=1.0), index)
