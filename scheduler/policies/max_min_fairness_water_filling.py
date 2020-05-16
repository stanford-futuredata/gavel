import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy


class MaxMinFairnessWaterFillingPolicyWithPerf(Policy):

    def __init__(self, priority_reweighting_policy):
        Policy.__init__(self, solver=None)
        self._name = 'MaxMinFairnessWaterFilling_Perf'
        self._proportional_policy = ProportionalPolicy()
        self._previous_priority_weights = None
        self._priority_reweighting_policy = priority_reweighting_policy

    def compute_priority_weights(self, priority_weights, entity_to_job_mapping,
                                 per_job_effective_throughputs):
        if self._priority_reweighting_policy is None:
            # Do nothing if reweighting policy is None.
            return priority_weights
        elif self._priority_reweighting_policy == 'multilevel_fairness':
            if entity_to_job_mapping is None:
                raise ValueError("entity_to_job_mapping cannot be None when "
                                 "priority_reweighting_policy is multilevel_fairness!")
            returned_priority_weights = {}
            for entity_id in entity_to_job_mapping:
                # The sum of final priority weights for all jobs in an entity
                # should be equal to the priority weight of that entity. The
                # final priority weight of a job should be proportional to the
                # passed-in weight. A job does not contribute its priority once
                # saturated.
                entity_weight = priority_weights[entity_id]
                total_job_priority_in_entity = 0.0
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in per_job_effective_throughputs:
                        continue
                    total_job_priority_in_entity += float(priority_weights[job_id])
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in per_job_effective_throughputs:
                        returned_priority_weights[job_id] = 0.0
                    else:
                        returned_priority_weights[job_id] = \
                            entity_weight * (float(priority_weights[job_id]) /
                                             total_job_priority_in_entity)
            return returned_priority_weights
        elif self._priority_reweighting_policy == 'fairness+fifo':
            if entity_to_job_mapping is None:
                raise ValueError("entity_to_job_mapping cannot be None when "
                                 "priority_reweighting_policy is fairness+fifo!")
            returned_priority_weights = {}
            for entity_id in entity_to_job_mapping:
                # Active jobs are given the corresponding entity's weight.
                # Jobs become active in FIFO order within an entity.
                entity_weight = priority_weights[entity_id]
                total_job_priority_in_entity = 0.0
                entity_to_job_mapping[entity_id].sort()
                done = False
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in per_job_effective_throughputs:
                        returned_priority_weights[job_id] = 0.0
                    elif not done:
                        returned_priority_weights[job_id] = entity_weight
                        done = True
                    else:
                        returned_priority_weights[job_id] = 0.0
            return returned_priority_weights
        else:
            raise ValueError("Unknown priority reweighting policy!")


    def _get_allocation(self, throughputs, index, priority_weights,
                        scale_factors_array, m, n,
                        per_job_effective_throughputs, best_c_so_far):
        x = cp.Variable(throughputs.shape)
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
        effective_throughputs = cp.sum(cp.multiply(throughputs, x), axis=1)

        # Solve max-min optimization problem over all jobs with priority
        # weight > 0.
        objective_terms = []
        c_terms_balance = np.zeros(m)
        for i, job_id in enumerate(job_ids):
            if job_id not in per_job_effective_throughputs:
                if priority_weights[job_ids.index(job_id)] > 0.0:
                    objective_term = scaled_effective_throughputs[job_id]
                    if self._previous_priority_weights is not None and \
                        self._previous_priority_weights[job_id] > 0:
                        objective_term -= best_c_so_far
                        c_terms_balance[i] = best_c_so_far
                    objective_terms.append(objective_term)
        if len(objective_terms) == 1:
            objective = cp.Maximize(objective_terms[0])
        else:
            objective = cp.Maximize(cp.minimum(*objective_terms))

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)

        # Add constraint for already saturated jobs.
        for i, job_id in enumerate(job_ids):
            if job_id in per_job_effective_throughputs:
                constraints.append(
                    effective_throughputs[i] ==
                    per_job_effective_throughputs[job_id])

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver='ECOS')

        if cvxprob.status != "optimal":
            raise Exception("WARNING: Non-optimal allocation in _get_allocation()!")

        return x, objective.value, objective.value + c_terms_balance

    def _get_bottleneck_jobs(self, throughputs, index, priority_weights,
                             scale_factors_array, m, n,
                             per_job_effective_throughputs,
                             c_terms):
        x = cp.Variable(throughputs.shape)
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
        effective_throughputs = cp.sum(cp.multiply(throughputs, x), axis=1)

        z = cp.Variable(m, boolean=True)
        objective = cp.Maximize(cp.sum(z))

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        for i, job_id in enumerate(job_ids):
            if job_id in per_job_effective_throughputs:
                constraints.append(
                    effective_throughputs[i] == per_job_effective_throughputs[job_id])
                constraints.append(z[i] == 0)
            else:
                if priority_weights[i] > 0.0:
                    constraints.append(
                        scaled_effective_throughputs[i] >= c_terms[i])
                    constraints.append(
                        (M * z[i]) >=
                            (scaled_effective_throughputs[i] - (c_terms[i] * 1.0001) + epsilon))
                    constraints.append(
                         (M * (1 - z[i])) >=
                            ((c_terms[i] * 1.0001) - scaled_effective_throughputs[i]))
                else:
                    constraints.append(z[i] == 0)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver='GLPK_MI')

        if cvxprob.status != "optimal":
            raise Exception("WARNING: Non-optimal allocation in _get_bottleneck_jobs()!")

        return x, z.value

    def get_allocation(self, original_unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, original_cluster_spec,
                       entity_to_job_mapping=None, verbose=False):
        unflattened_throughputs = copy.deepcopy(original_unflattened_throughputs)        
        cluster_spec = copy.deepcopy(original_cluster_spec)

        done = False
        original_throughputs, original_index = super().flatten(original_unflattened_throughputs,
                                                               cluster_spec)
        (original_job_ids, original_worker_types) = original_index
        per_job_effective_throughputs = {}
        num_iterations = 0
        c = 0
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

            priority_weights = self.compute_priority_weights(
                unflattened_priority_weights,
                entity_to_job_mapping,
                per_job_effective_throughputs)
            previous_priority_weights = copy.copy(priority_weights)
            if verbose:
                print("Using the following as priority weights:", np.array(
                    [priority_weights[job_id] for job_id in job_ids]))
            priority_weights = np.array(
                [1. / priority_weights[job_id] if priority_weights[job_id] > 0 else 0.0
                 for job_id in job_ids])

            proportional_throughputs = self._proportional_policy.get_throughputs(
                throughputs, index, cluster_spec)
            priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                           1.0 / proportional_throughputs.reshape((m, 1)))

            x_returned, c_returned, c_terms = self._get_allocation(
                throughputs, index, priority_weights, scale_factors_array,
                m, n, per_job_effective_throughputs=per_job_effective_throughputs,
                best_c_so_far=c)
            self._previous_priority_weights = previous_priority_weights
            if num_iterations == 0:
                print("Objective value: %.3f" % c_returned)

            # Find bottleneck job_ids.
            _, z = self._get_bottleneck_jobs(
                throughputs, index, priority_weights, scale_factors_array, m, n,
                per_job_effective_throughputs=per_job_effective_throughputs,
                c_terms=c_terms)
            old_len_effective_throughputs = len(per_job_effective_throughputs)
            for i, job_id in enumerate(job_ids):
                if job_id not in per_job_effective_throughputs and (z is None or not z[i]) \
                    and priority_weights[i] > 0.0:
                    print("Iteration %d:" % num_iterations, job_id)
                    per_job_effective_throughputs[job_id] = c_terms[i] / (
                        priority_weights[i][0] * scale_factors[job_id])
            x = x_returned
            c += c_returned
            if old_len_effective_throughputs == len(per_job_effective_throughputs):
                done = True
            if verbose:
                print("At the end of iteration %d:" % num_iterations,
                    x.value)
            num_iterations += 1
            if len(unflattened_throughputs) == len(per_job_effective_throughputs):
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
        effective_throughputs = np.sum(np.multiply(
            original_throughputs, x.value), axis=1)

        print("Effective throughputs:", effective_throughputs)
        print("Final objective: %.3f" % np.min(scaled_effective_throughputs))
        print("Constraints:",
            np.multiply(x.value, scale_factors_array).sum(axis=0),
            x.value.sum(axis=1))

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
