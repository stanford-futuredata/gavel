import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
import max_min_fairness

class FinishTimeFairnessPolicy(Policy):

    def __init__(self, solver):
        self._name = 'FinishTimeFairness'
        self._finish_time_fairness_perf_policy = \
            FinishTimeFairnessPolicyWithPerf(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = 1.0

        return self._finish_time_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors,
            unflattened_priority_weights,
            times_since_start,
            num_steps_remaining, cluster_spec)


class FinishTimeFairnessPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'FinishTimeFairness_Perf'
        self._isolated_policy = max_min_fairness.MaxMinFairnessPolicy(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]

        # TODO: Do something with these priority_weights.
        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        # Create allocation variable, and isolated allocation.
        x = cp.Variable(throughputs.shape)
        x_isolated_dict = self._isolated_policy.get_allocation(
            unflattened_throughputs, scale_factors, unflattened_priority_weights,
            cluster_spec)
        x_isolated = np.zeros(throughputs.shape)
        for i in range(m):
            for j in range(n):
                x_isolated[(i, j)] = x_isolated_dict[job_ids[i]][worker_types[j]]

        isolated_throughputs = np.sum(np.multiply(throughputs, x_isolated),
                                      axis=1)
        expected_time_fractions = []
        for i in range(len(job_ids)):
            allocation_throughput = cp.sum(cp.multiply(throughputs[i], x[i]))
            expected_time_isolated = times_since_start[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] / isolated_throughputs[i])
            expected_time_allocation = times_since_start[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] * cp.inv_pos(allocation_throughput))
            expected_time_fraction = expected_time_allocation / expected_time_isolated
            expected_time_fractions.append(expected_time_fraction)
        if len(expected_time_fractions) == 1:
            objective = cp.Minimize(expected_time_fractions[0])
        else:
            objective = cp.Minimize(cp.maximum(*expected_time_fractions))

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


class FinishTimeFairnessPolicyWithPacking(PolicyWithPacking):

    def __init__(self, solver):
        PolicyWithPacking.__init__(self, solver)
        self._name = 'FinishTimeFairness_Packing'
        self._isolated_policy = max_min_fairness.MaxMinFairnessPolicy(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        all_throughputs, index = \
            self.flatten(d=unflattened_throughputs,
                         cluster_spec=cluster_spec,
                         priority_weights=unflattened_priority_weights)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        x = cp.Variable((m, n))

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
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

        unflattened_throughputs_no_packed_jobs = {}
        for single_job_id in single_job_ids:
            unflattened_throughputs_no_packed_jobs[single_job_id] = {}
            for worker_type in worker_types:
                unflattened_throughputs_no_packed_jobs[single_job_id][worker_type] = 1.0
        x_isolated_dict = self._isolated_policy.get_allocation(
            unflattened_throughputs_no_packed_jobs, scale_factors,
            unflattened_priority_weights, cluster_spec)
        x_isolated = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if not job_ids[i].is_pair():
                    x_isolated[(i, j)] = x_isolated_dict[job_ids[i]][worker_types[j]]

        single_throughputs = np.zeros((len(single_job_ids), n))
        expected_time_fractions = []
        for i in range(len(all_throughputs)):
            indexes = relevant_combinations[single_job_ids[i]]
            isolated_throughput = np.sum(np.multiply(
                all_throughputs[i][indexes],
                x_isolated[indexes]))
            allocation_throughput = cp.sum(cp.multiply(
                all_throughputs[i][indexes],
                x[indexes]))
            expected_time_isolated = times_since_start[single_job_ids[i]] + \
                (num_steps_remaining[single_job_ids[i]] / isolated_throughput)
            expected_time_allocation = times_since_start[single_job_ids[i]] + \
                (num_steps_remaining[single_job_ids[i]] * cp.inv_pos(allocation_throughput))
            expected_time_fraction = expected_time_allocation / expected_time_isolated
            expected_time_fractions.append(expected_time_fraction)
        if len(expected_time_fractions) == 1:
            objective = cp.Minimize(expected_time_fractions[0])
        else:
            objective = cp.Minimize(cp.maximum(*expected_time_fractions))

        # Make sure the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, single_job_ids,
                                                scale_factors_array,
                                                relevant_combinations)

        # Explicitly constrain all allocation values with an effective scale
        # factor of 0 to be 0.
        # NOTE: This is not strictly necessary because these allocation values
        # do not affect the optimal allocation for nonzero scale factor
        # combinations.
        for i in range(m):
            for j in range(n):
                if scale_factors_array[i,j] == 0:
                    constraints.append(x[i,j] == 0)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return self.unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
