import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class MinTotalDurationPolicy(Policy):

    def __init__(self, solver):
        self._name = 'MinTotalDuration'
        self._min_total_duration_perf_policy = \
            MinTotalDurationPolicyWithPerf(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       num_steps_remaining, cluster_spec):
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

        return self._min_total_duration_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, num_steps_remaining,
            cluster_spec)


class MinTotalDurationPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MinTotalDuration_Perf'

    def get_allocation_helper(self, throughputs, scale_factors_array, T):
        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(1)
        # Make sure the allocation can fit in the cluster, and that the
        # currently active jobs can finish in time T.
        constraints = self.get_base_constraints(x, scale_factors_array)
        constraints.append(
            cp.sum(cp.multiply(throughputs, x), axis=1) >= (
                self._num_steps_remaining / T)
        )
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
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

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
        constraints = self.get_base_constraints(x, single_job_ids,
                                                scale_factors_array,
                                                relevant_combinations)

        # See if passed in T is feasible.
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
                                                 cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        if index is None: return None
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        self._num_steps_remaining = [num_steps_remaining[single_job_id]
                                     for single_job_id in single_job_ids]

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        (m, n) = all_throughputs[0].shape
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

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
