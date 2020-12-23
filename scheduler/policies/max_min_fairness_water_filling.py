import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy


class WaterFillingAlgorithm:

    def __init__(self, priority_reweighting_policies):
        self._previous_priority_weights = None
        self._priority_reweighting_policies = priority_reweighting_policies
        self._lp = None
        self._milp = None

    def _compute_priority_weights(self, entity_weights, priority_weights, entity_to_job_mapping,
                                  final_normalized_effective_throughputs, job_ids):
        returned_priority_weights = {}
        if self._priority_reweighting_policies is None:
            # Do nothing if reweighting policy is None.
            return priority_weights
        if entity_to_job_mapping is None:
            raise ValueError("entity_to_job_mapping cannot be None when "
                             "priority_reweighting_policies is not None!")
        for entity_id in entity_to_job_mapping:
            priority_reweighting_policy = self._priority_reweighting_policies[
                entity_id]
            if priority_reweighting_policy == 'fairness':
                # The sum of final priority weights for all jobs in an entity
                # should be equal to the priority weight of that entity. The
                # final priority weight of a job should be proportional to the
                # passed-in weight. A job does not contribute its priority once
                # saturated.
                entity_weight = entity_weights[entity_id]
                total_job_priority_in_entity = 0.0
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in final_normalized_effective_throughputs:
                        continue
                    total_job_priority_in_entity += float(priority_weights[job_id])
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in final_normalized_effective_throughputs:
                        returned_priority_weights[job_id] = 0.0
                    else:
                        returned_priority_weights[job_id] = \
                            entity_weight * (float(priority_weights[job_id]) /
                                             total_job_priority_in_entity)
            elif priority_reweighting_policy == 'fifo':
                # Active jobs are given the corresponding entity's weight.
                # Jobs become active in FIFO order within an entity.
                entity_weight = entity_weights[entity_id]
                total_job_priority_in_entity = 0.0
                entity_to_job_mapping[entity_id].sort()
                done = False
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in final_normalized_effective_throughputs:
                        returned_priority_weights[job_id] = 0.0
                    elif not done:
                        returned_priority_weights[job_id] = entity_weight
                        done = True
                    else:
                        returned_priority_weights[job_id] = 0.0
            else:
                raise ValueError("Unknown priority reweighting policy!")
        return returned_priority_weights


    def _get_allocation(self, job_ids, priority_weights,
                        proportional_throughputs,
                        scale_factors_array, m, n,
                        final_normalized_effective_throughputs,
                        normalized_effective_throughputs_so_far):
        M = self._M

        if self._lp is None:
            self._lp_variables_and_parameters = {}
            self._lp_variables_and_parameters['x'] = cp.Variable((m, n))
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter'] = cp.Parameter(len(job_ids))
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter'] = cp.Parameter(len(job_ids))
            self._lp_variables_and_parameters[
                'multiplicative_terms_parameter'] = cp.Parameter(len(job_ids))
            self._lp_variables_and_parameters[
                'additive_terms_parameter'] = cp.Parameter(len(job_ids))

        x = self._lp_variables_and_parameters['x']
        normalized_effective_throughputs_so_far_parameter = \
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter']
        normalized_effective_throughputs_lower_bounds_parameter = \
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter']
        multiplicative_terms_parameter = \
            self._lp_variables_and_parameters['multiplicative_terms_parameter']
        additive_terms_parameter = \
            self._lp_variables_and_parameters['additive_terms_parameter']

        effective_throughputs = self._get_effective_throughputs(x)
        normalized_effective_throughputs = cp.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(len(job_ids)))

        # Solve max-min optimization problem over all jobs with priority
        # weight > 0.
        normalized_effective_throughputs_so_far_parameter.value = \
            normalized_effective_throughputs_so_far
        objective_terms = normalized_effective_throughputs - \
            normalized_effective_throughputs_so_far_parameter
        multiplicative_terms = np.zeros(len(job_ids))
        mask = np.zeros(len(job_ids))
        additive_terms = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id not in final_normalized_effective_throughputs:
                if priority_weights[i] > 0.0:
                    multiplicative_terms[i] = priority_weights[i] * scale_factors_array[i, 0]
                    mask[i] = 1.0 / multiplicative_terms[i]
                else:
                    additive_terms[i] = M
            else:
                additive_terms[i] = M
        multiplicative_terms_parameter.value = multiplicative_terms
        additive_terms_parameter.value = additive_terms

        normalized_effective_throughputs_lower_bounds = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id in final_normalized_effective_throughputs:
                normalized_effective_throughputs_lower_bounds[i] = \
                    final_normalized_effective_throughputs[job_id]
            else:
                normalized_effective_throughputs_lower_bounds[i] = \
                    normalized_effective_throughputs_so_far[i]
        normalized_effective_throughputs_lower_bounds_parameter.value = \
            normalized_effective_throughputs_lower_bounds

        if self._lp is None:
            self._lp_objective = cp.Maximize(cp.min(
                cp.multiply(objective_terms, multiplicative_terms_parameter) +
                additive_terms_parameter))
            # Specify constraints.
            constraints = self._get_constraints(x, scale_factors_array)
            constraints.append(normalized_effective_throughputs >=
                normalized_effective_throughputs_lower_bounds_parameter)

            self._lp = cp.Problem(self._lp_objective, constraints)

        result = self._lp.solve(solver='ECOS', warm_start=True)

        return x.value, self._lp_objective.value, mask

    def _get_bottleneck_jobs(self, job_ids, priority_weights,
                             proportional_throughputs,
                             scale_factors_array, m, n,
                             final_normalized_effective_throughputs,
                             normalized_effective_throughputs_so_far, slack=1.0001):
        M = self._M
        epsilon = 1e-5

        if self._milp is None:
            self._milp_variables_and_parameters = {}
            self._milp_variables_and_parameters['x'] = cp.Variable((m, n))
            self._milp_variables_and_parameters['z'] = cp.Variable(len(job_ids), boolean=True)
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter'] = cp.Parameter(len(job_ids))
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter'] = cp.Parameter(len(job_ids))
            self._milp_variables_and_parameters['mask_parameter'] = cp.Parameter(len(job_ids))

        x = self._milp_variables_and_parameters['x']
        z = self._milp_variables_and_parameters['z']
        normalized_effective_throughputs_so_far_parameter = \
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter']
        normalized_effective_throughputs_lower_bounds_parameter = \
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter']
        mask_parameter = self._milp_variables_and_parameters['mask_parameter']

        effective_throughputs = self._get_effective_throughputs(x)
        normalized_effective_throughputs = cp.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(len(job_ids)))

        normalized_effective_throughputs_so_far_parameter.value = \
            normalized_effective_throughputs_so_far

        normalized_effective_throughputs_lower_bounds = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id in final_normalized_effective_throughputs:
                normalized_effective_throughputs_lower_bounds[i] = \
                    final_normalized_effective_throughputs[job_id]
            else:
                normalized_effective_throughputs_lower_bounds[i] = \
                    normalized_effective_throughputs_so_far[i]
        normalized_effective_throughputs_lower_bounds_parameter.value = \
            normalized_effective_throughputs_lower_bounds

        mask = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id in final_normalized_effective_throughputs:
                mask[i] = 1.0
            else:
                if priority_weights[i] == 0.0:
                    mask[i] = 1.0
        mask_parameter.value = mask

        if self._milp is None:
            objective = cp.Maximize(cp.sum(z))
            # Specify constraints.
            constraints = self._get_constraints(x, scale_factors_array)
            constraints.append(
                (M * z) >=
                    (normalized_effective_throughputs -
                     (normalized_effective_throughputs_so_far_parameter * slack) +
                     epsilon))
            constraints.append(
                 (M * (1 - z)) >=
                    ((normalized_effective_throughputs_so_far_parameter * slack) -
                     normalized_effective_throughputs))
            constraints.append(normalized_effective_throughputs >=
                normalized_effective_throughputs_lower_bounds_parameter)
            constraints.append(cp.multiply(mask_parameter, z) == 0)

            self._milp = cp.Problem(objective, constraints)

        result = self._milp.solve(solver='GLPK_MI', warm_start=True)
        if self._milp.status != "optimal":
            raise Exception("WARNING: Non-optimal allocation in _get_bottleneck_jobs()!")

        return z.value

    def _run_get_allocation_iterations(self, job_ids, m, n,
                                       proportional_throughputs,
                                       scale_factors_array,
                                       entity_weights,
                                       unflattened_priority_weights, cluster_spec,
                                       entity_to_job_mapping, verbose):
        done = False

        final_normalized_effective_throughputs = {}
        normalized_effective_throughputs_so_far = np.zeros(len(job_ids))
        num_iterations = 0
        c, x, mask = 0, None, None
        while not done:
            priority_weights = self._compute_priority_weights(
                entity_weights,
                unflattened_priority_weights,
                entity_to_job_mapping,
                final_normalized_effective_throughputs, job_ids)
            previous_priority_weights = copy.copy(priority_weights)
            if verbose:
                print("Using the following as priority weights:", np.array(
                    [priority_weights[job_id] for job_id in job_ids]))
            priority_weights = np.array(
                [1. / priority_weights[job_id] if priority_weights[job_id] > 0 else 0.0
                 for job_id in job_ids])

            old_x, old_c, old_mask = np.copy(x), c, np.copy(mask)
            try:
                x, c, mask = self._get_allocation(
                    job_ids, priority_weights, proportional_throughputs,
                    scale_factors_array,
                    m, n, final_normalized_effective_throughputs,
                    normalized_effective_throughputs_so_far)
                if x is None:
                    x, c, mask = old_x, old_c, old_mask
                    done = True
                else:
                    normalized_effective_throughputs_so_far += np.multiply(mask, c)
            except:
                x, c, mask = old_x, old_c, old_mask
                done = True

            self._previous_priority_weights = previous_priority_weights
            if verbose:
                if num_iterations == 0:
                    print("Objective value: %.3f" % c)

            # Find bottleneck job_ids.
            try:
                z = self._get_bottleneck_jobs(
                    job_ids, priority_weights,
                    proportional_throughputs,
                    scale_factors_array, m, n,
                    final_normalized_effective_throughputs,
                    normalized_effective_throughputs_so_far)
            except Exception as e:
                z = np.zeros(len(job_ids))
            old_len_effective_throughputs = len(final_normalized_effective_throughputs)
            for i, job_id in enumerate(job_ids):
                if job_id not in final_normalized_effective_throughputs and (z is None or not z[i]) \
                    and priority_weights[i] > 0.0:
                    if verbose:
                        print("Iteration %d:" % num_iterations, job_id)
                    final_normalized_effective_throughputs[job_id] = \
                        normalized_effective_throughputs_so_far[i]
            if old_len_effective_throughputs == len(final_normalized_effective_throughputs):
                done = True
            num_iterations += 1
            if len(final_normalized_effective_throughputs) == m:
                done = True
        if verbose:
            print("Number of iterations: %d" % num_iterations)

        return x


class MaxMinFairnessWaterFillingPolicy(Policy, WaterFillingAlgorithm):

    def __init__(self, priority_reweighting_policies=None):
        self._name = 'MaxMinFairnessWaterFilling'
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessWaterFillingPolicyWithPerf(priority_reweighting_policies)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       entity_weights=None,
                       entity_to_job_mapping=None, verbose=False,
                       return_effective_throughputs=False):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index
        m, n = len(job_ids), len(worker_types)

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = 1.0

        unflattened_x = \
            self._max_min_fairness_perf_policy.get_allocation(
                new_unflattened_throughputs, scale_factors,
                unflattened_priority_weights,
                cluster_spec,
                entity_weights=entity_weights,
                entity_to_job_mapping=entity_to_job_mapping,
                verbose=verbose,
                return_effective_throughputs=False)
        x = np.zeros((len(job_ids), len(worker_types)))
        for i, job_id in enumerate(job_ids):
            for j, worker_type in enumerate(worker_types):
                x[i, j] = unflattened_x[job_id][worker_type]
        if return_effective_throughputs:
            effective_throughputs = np.sum(np.multiply(
                throughputs, x), axis=1)
            proportional_throughputs = \
                self._max_min_fairness_perf_policy._proportional_policy.get_throughputs(
                    throughputs, index, cluster_spec)
            normalized_effective_throughputs = np.multiply(
                effective_throughputs,
                1.0 / proportional_throughputs.reshape(m))
            return normalized_effective_throughputs, job_ids
        return unflattened_x

class MaxMinFairnessWaterFillingPolicyWithPerf(Policy, WaterFillingAlgorithm):

    def __init__(self, priority_reweighting_policies=None):
        WaterFillingAlgorithm.__init__(self, priority_reweighting_policies)
        Policy.__init__(self, solver=None)
        self._name = 'MaxMinFairnessWaterFilling_Perf'
        self._proportional_policy = ProportionalPolicy()

    def _get_constraints(self, x, scale_factors_array):
        return self.get_base_constraints(x, scale_factors_array)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       entity_weights=None,
                       entity_to_job_mapping=None, verbose=False,
                       return_effective_throughputs=False):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None

        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)

        self._M = np.max(
            np.multiply(throughputs * (1.0 / proportional_throughputs).reshape((m, 1)),
                        scale_factors_array))
        self._get_effective_throughputs = lambda x: \
            cp.sum(cp.multiply(throughputs, x), axis=1)

        x = self._run_get_allocation_iterations(
            job_ids, m, n, proportional_throughputs,
            scale_factors_array,
            entity_weights,
            unflattened_priority_weights, cluster_spec,
            entity_to_job_mapping=entity_to_job_mapping, verbose=verbose)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])
        priority_weights = np.multiply(
            priority_weights.reshape((m, 1)),
            1.0 / proportional_throughputs.reshape((m, 1)))
        effective_throughputs = np.sum(np.multiply(
            throughputs, x), axis=1)
        normalized_effective_throughputs = np.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(m))

        if verbose:
            print("Normalized effective throughputs:",
                normalized_effective_throughputs)
            print("Constraints:",
                np.multiply(x, scale_factors_array).sum(axis=0),
                x.sum(axis=1))

        if return_effective_throughputs:
            return normalized_effective_throughputs, job_ids

        self._lp = None
        self._milp = None

        return super().unflatten(x.clip(min=0.0).clip(max=1.0), index)


class MaxMinFairnessWaterFillingPolicyWithPacking(PolicyWithPacking, WaterFillingAlgorithm):

    def __init__(self, priority_reweighting_policies=None):
        WaterFillingAlgorithm.__init__(self, priority_reweighting_policies)
        PolicyWithPacking.__init__(self, solver=None)
        self._name = 'MaxMinFairnessWaterFilling_Packing'
        self._proportional_policy = ProportionalPolicy()

    def _get_constraints(self, x, scale_factors_array):
        return self.get_base_constraints(x, self._single_job_ids,
                                         scale_factors_array,
                                         self._relevant_combinations)

    def _get_M(self, all_throughputs, index, proportional_throughputs,
               scale_factors_array):
        (_, single_job_ids, _, relevant_combinations) = index
        max_throughputs = []
        for i, single_job_id in enumerate(single_job_ids):
            indexes = relevant_combinations[single_job_id]
            max_throughputs.append(np.max(
                np.multiply(all_throughputs[i][indexes], scale_factors_array[indexes])))
            max_throughputs[-1] /= proportional_throughputs[i]
        return np.max(np.array(max_throughputs))

    def _get_effective_throughputs_helper(self, x, all_throughputs, index):
        (_, single_job_ids, _, relevant_combinations) = index
        effective_throughputs = []
        for i, single_job_id in enumerate(single_job_ids):
            indexes = relevant_combinations[single_job_id]
            effective_throughputs.append(cp.sum(cp.multiply(
                all_throughputs[i][indexes],
                x[indexes])))
        return cp.hstack(effective_throughputs)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       entity_weights=None,
                       entity_to_job_mapping=None, verbose=False,
                       return_effective_throughputs=False):
        all_throughputs, index = \
            self.flatten(d=unflattened_throughputs,
                         cluster_spec=cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None

        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        self._single_job_ids = single_job_ids
        self._relevant_combinations = relevant_combinations
        (m, n) = all_throughputs[0].shape

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)
        throughputs_no_packed_jobs = np.zeros((len(single_job_ids), n))
        for i, single_job_id in enumerate(single_job_ids):
            for j, worker_type in enumerate(worker_types):
                throughputs_no_packed_jobs[i, j] = \
                    unflattened_throughputs[single_job_id][worker_type]
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs_no_packed_jobs,
            (single_job_ids, worker_types),
            cluster_spec)

        self._M = self._get_M(all_throughputs, index, proportional_throughputs,
                              scale_factors_array)
        self._get_effective_throughputs = lambda x: \
            self._get_effective_throughputs_helper(x, all_throughputs, index)

        x = self._run_get_allocation_iterations(
            single_job_ids, m, n, proportional_throughputs,
            scale_factors_array,
            entity_weights,
            unflattened_priority_weights, cluster_spec,
            entity_to_job_mapping=entity_to_job_mapping, verbose=verbose)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[single_job_id]
             for single_job_id in single_job_ids])
        priority_weights = np.multiply(
            priority_weights.reshape((len(single_job_ids), 1)),
            1.0 / proportional_throughputs.reshape((len(single_job_ids), 1)))
        effective_throughputs = np.zeros(len(single_job_ids))
        for i, single_job_id in enumerate(single_job_ids):
            indexes = relevant_combinations[single_job_id]
            effective_throughputs[i] = np.sum(np.multiply(
                all_throughputs[i][indexes],
                x[indexes]))
        normalized_effective_throughputs = np.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(len(single_job_ids)))

        if verbose:
            print("Normalized effective throughputs:",
                normalized_effective_throughputs)
            print("Constraints:",
                np.multiply(x, scale_factors_array).sum(axis=0),
                x.sum(axis=1))

        if return_effective_throughputs:
            return normalized_effective_throughputs, single_job_ids

        self._lp = None
        self._milp = None

        return super().unflatten(x.clip(min=0.0).clip(max=1.0), index)
