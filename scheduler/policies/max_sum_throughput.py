import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class ThroughputSumWithPerf(Policy):

    def __init__(self, solver):
        self._name = 'ThroughputSumWithPerf'
        self._policy = ThroughputNormalizedByCostSumWithPerfSLOs(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec):
        return self._policy.get_allocation(unflattened_throughputs,
                                           scale_factors,
                                           cluster_spec)

class ThroughputNormalizedByCostSumWithPerf(Policy):
    def __init__(self, solver):
        self._name = 'ThroughputNormalizedByCostSum_Perf'
        self._policy = ThroughputNormalizedByCostSumWithPerfSLOs(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec, instance_costs):
        return self._policy.get_allocation(unflattened_throughputs,
                                           scale_factors,
                                           cluster_spec,
                                           instance_costs=instance_costs)

class ThroughputNormalizedByCostSumWithPerfSLOs(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'ThroughputNormalizedByCostSum_PerfSLOs'

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec, instance_costs=None, SLOs={},
                       num_steps_remaining={}):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale = 1.0 / throughputs.sum(axis=1)

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        x = cp.Variable(throughputs.shape)
        instance_costs_array = np.ones((1, n))
        if instance_costs is not None:
            for i in range(n):
                instance_costs_array[0, i] = instance_costs[worker_types[i]]
        objective = \
            cp.Maximize(cp.sum(cp.sum(cp.multiply(throughputs /
                                                  instance_costs_array, x),
                                      axis=1)))

        # Make sure that a given job is not over-allocated resources.
        constraints = self.get_base_constraints(x, scale_factors_array)
        SLO_constraints = []
        for job_id in SLOs:
            i = job_ids.index(job_id)
            assert(job_id in num_steps_remaining)
            SLO_constraints.append(
                cp.sum(cp.multiply(throughputs[i], x[i])) >=
                    (num_steps_remaining[job_id] / SLOs[job_id])
            )
        cvxprob = cp.Problem(objective, constraints + SLO_constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        if x.value is None:
            print('WARNING: No allocation possible with provided SLOs!')
            cvxprob = cp.Problem(objective, constraints)
            result = cvxprob.solve(solver=self._solver)

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)

class ThroughputNormalizedByCostSumWithPackingSLOs(PolicyWithPacking):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'ThroughputNormalizedByCostSum_PackingSLOs'

    def get_allocation(self, unflattened_throughputs, scale_factors, cluster_spec,
                       instance_costs=None, SLOs={}, num_steps_remaining={}):
        all_throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        x = cp.Variable((m, n))
        instance_costs_array = np.ones((m, n))
        if instance_costs is not None:
            for i in range(m):
                for j in range(n):
                    instance_costs_array[i,j] = \
                        instance_costs[worker_types[j]]

        objective_terms = []
        for i in range(len(single_job_ids)):
            indexes = relevant_combinations[single_job_ids[i]]
            objective_terms.append(cp.sum(cp.multiply(
                all_throughputs[i][indexes] /\
                    instance_costs_array[indexes], x[indexes])))

        if len(objective_terms) == 1:
            objective = cp.Maximize(objective_terms[0])
        else:
            objective = cp.Maximize(cp.sum(cp.hstack(objective_terms)))

        # Make sure that a given job is not over-allocated resources.
        constraints = self.get_base_constraints(x, single_job_ids,
                                                scale_factors_array,
                                                relevant_combinations)

        SLO_constraints = []
        per_job_throughputs = []
        per_job_SLOs = []
        for job_id in SLOs:
            i = job_ids.index(job_id)
            assert(job_id in num_steps_remaining)
            indexes = relevant_combinations[single_job_ids[i]]
            throughput = cp.sum(cp.multiply(
                all_throughputs[i][indexes], x[indexes]))
            per_job_throughputs.append(throughput)
            per_job_SLOs.append(num_steps_remaining[job_id] / SLOs[job_id])
        if len(per_job_throughputs) > 0:
            SLO_constraints.append(cp.vstack(per_job_throughputs) >=
                                   cp.vstack(per_job_SLOs))
        cvxprob = cp.Problem(objective, constraints + SLO_constraints)
        result = cvxprob.solve(solver=self._solver)

        if x.value is None:
            print('WARNING: No allocation possible with provided SLOs!')
            cvxprob = cp.Problem(objective, constraints)
            result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
