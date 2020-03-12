import copy
import cvxpy as cp
import numpy as np
import random

import job_id_pair

class Policy:

    def __init__(self, solver='ECOS'):
        self._name = None
        self._solver = solver

    @property
    def name(self):
        return self._name

    def flatten(self, d, cluster_spec):
        """Converts a 2-level dict to a NumPy array."""

        job_ids = sorted(list(d.keys()))
        if len(job_ids) == 0:
            return None, None
        worker_types = sorted(list(d[job_ids[0]].keys()))
        self._num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]
        if len(worker_types) == 0:
            return None, None
        m = []
        for job_id in job_ids:
            m_row = []
            for worker_type in worker_types:
                m_row.append(d[job_id][worker_type])
            m.append(m_row)
        return np.array(m), (job_ids, worker_types)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_ids, worker_types) = index
        d = {}
        for i in range(len(job_ids)):
            d[job_ids[i]] = {}
            for j in range(len(worker_types)):
                d[job_ids[i]][worker_types[j]] = m[i][j]
        return d


class PolicyWithPacking(Policy):

    def __init__(self, solver='ECOS'):
        Policy.__init__(self, solver)

    def flatten(self, d, cluster_spec, normalize=True,
                priority_weights=None):
        """
        Converts a 2-level dict to a NumPy array.

        Job ID combinations in the input dict are either a tuple or an integer.
        If a tuple, represents a combination run on a GPU concurrently.
        If an integer, represents a single job / application run on the
        GPU.

        Returns a list of each user's normalized throughput matrix and an
        index to reconstruct the allocation as a dict.
        """
        job_ids = sorted(list(d.keys()))
        if len(job_ids) == 0:
            return None, None, None
        worker_types = sorted(list(d[job_ids[0]].keys()))
        self._num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]

        # Stores which indexes in job_ids are relevant for each single job ID.
        relevant_combinations = {}
        single_job_ids = set()
        sorted_single_job_ids = []
        for i, job_id in enumerate(job_ids):
            if not job_id.is_pair():
                single_job_ids.add(job_id)
                sorted_single_job_ids.append(job_id)
                if job_id not in relevant_combinations:
                    relevant_combinations[job_id] = []
                relevant_combinations[job_id].append(i)
            else:
                for single_job_id in job_id.singletons():
                    if single_job_id not in relevant_combinations:
                        relevant_combinations[single_job_id] = []
                    relevant_combinations[single_job_id].append(i)

        # Compute normalizing factor for each individual job, this normalizing
        # factor will be used to normalize throughputs for the same job in job
        # combinations as well.
        if normalize:
            normalizing_factors = {}
            for single_job_id in single_job_ids:
                normalizing_factor = 0.0
                for worker_type in worker_types:
                    normalizing_factor += d[single_job_id][worker_type]
                normalizing_factors[single_job_id] = normalizing_factor

        if len(worker_types) == 0:
            return None, None, None

        shape = (len(single_job_ids), len(job_ids), len(worker_types))
        all_m = np.zeros(shape, dtype=np.float32)
        # Compute the throughput matrix for each individual job.
        for i, single_job_id in enumerate(sorted_single_job_ids):
            # Each throughput matrix has dimension
            # (num_app_combinations x num_worker_types).
            for j in relevant_combinations[single_job_id]:
                job_id = job_ids[j]
                for k, worker_type in enumerate(worker_types):
                    # If job ID of interest is not in this job_id_combination,
                    # throughput should be 0.
                    # Otherwise, use the right throughput from the input dict.
                    if job_id in single_job_ids:
                        if job_id == single_job_id:
                            all_m[i][j][k] = d[job_id][worker_type]
                    else:
                        if single_job_id.overlaps_with(job_id):
                            # Find the index of the job of interest in the job
                            # combination tuple.
                            index = job_id.as_tuple().index(single_job_id[0])
                            throughputs = d[job_id][worker_type]
                            all_m[i][j][k] = d[job_id][worker_type][index]
            # Normalize.
            if normalize:
                all_m[i] /= normalizing_factors[single_job_id]
                if priority_weights is not None:
                    all_m[i] /= priority_weights[single_job_id]
        return all_m, (job_ids, sorted_single_job_ids, worker_types,
                       relevant_combinations)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_id_combinations, single_job_ids, worker_types, _) = index
        d = {}
        for i in range(len(job_id_combinations)):
            d[job_id_combinations[i]] = {}
            for j in range(len(worker_types)):
                d[job_id_combinations[i]][worker_types[j]] = m[i][j]
        return d


class MaxMinFairnessPolicy(Policy):

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


class MaxMinFairnessPolicyWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MaxMinFairness_Perf'

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
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

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        scale = 1.0 / throughputs.sum(axis=1)
        throughputs = throughputs * scale.reshape(m, 1)
        throughputs = throughputs * priority_weights.reshape((m, 1))

        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.min(cp.sum(cp.multiply(
                np.multiply(throughputs, scale_factors_array), x), axis=1)))
        # Make sure that the allocation can fit in the cluster.
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= self._num_workers,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


class MaxMinFairnessPolicyWithPacking(PolicyWithPacking):

    def __init__(self, solver):
        PolicyWithPacking.__init__(self, solver)
        self._name = 'MaxMinFairness_Packing'

    def get_allocation_v3(self, unflattened_job_type_throughputs,
                          job_id_to_job_type, scale_factors,
                          priority_weights, cluster_spec):
        # TODO: Rename unflattened_job_type_throughputs ->
        #       unflattened_throughputs
        job_ids = sorted(job_id_to_job_type.keys())
        job_types = sorted(unflattened_job_type_throughputs.keys())
        worker_types = sorted(cluster_spec.keys())
        num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]

        # Create a map from job type to list of job indexes.
        job_type_to_job_idx = {}
        for i, job_id in enumerate(job_ids):
            job_type = job_id_to_job_type[job_id]
            if job_type not in job_type_to_job_idx:
                job_type_to_job_idx[job_type] = []
            job_type_to_job_idx[job_type].append(i)

        # Num jobs.
        n = len(job_ids)
        # Num job_types.
        a = len(unflattened_job_type_throughputs.keys())
        # Num worker_types.
        m = len(worker_types)
        # Num varibles per job.
        num_vars_per_job = 1 + a

        # Compute normalizing factor for each job type.
        # This normalizing factor will be used to normalize throughputs for the
        # same job type in job-job type combinations, as well.
        normalizing_factors = {}
        for job_type in job_types:
            unflattened_throughputs = \
                unflattened_job_type_throughputs[job_type]
            normalizing_factor = 0.0
            for worker_type in worker_types:
                normalizing_factor += \
                    unflattened_throughputs[worker_type][None]
            normalizing_factors[job_type] = normalizing_factor

        # Set up flattened job type throughputs.
        flattened_job_type_throughputs = np.zeros(shape=(a, (1 + a) * m),
                                                  dtype=np.float32)
        for i, job_type in enumerate(job_types):
            unflattened_throughputs = \
                unflattened_job_type_throughputs[job_type]
            for k, worker_type in enumerate(worker_types):
                for j, other_job_type in enumerate([None] + job_types):
                    flattened_job_type_throughputs[i,k*(1+a)+j] = \
                        unflattened_throughputs[worker_type][other_job_type]
        for i, job_type in enumerate(job_types):
            flattened_job_type_throughputs[i] /= normalizing_factors[job_type]

        # Set up scale factors.
        flattened_scale_factors = \
            np.reshape([scale_factors[job_id] for job_id in job_ids], (n, 1))
        scale_factors_array = np.tile(flattened_scale_factors,
                                        (1, num_vars_per_job * m))

        # Set up masks to avoid double-counting allocation values when
        # computing constraint that the sum of allocation values of each
        # worker type must be <= the number of workers of that worker type.
        # TODO: Change this if we ever consider combinations larger than pairs.
        masks = np.full(shape=(n, num_vars_per_job), fill_value=0.5)
        masks[:,0] = 1.0

        # Allocation matrix.
        x = cp.Variable((n, num_vars_per_job * m))

        constraints = [
            # All allocation values must be >= 0.
            x >= 0,
            # The sum of allocation values for each job must be <= 1.
            cp.sum(x, axis=1) <= 1
        ]

        # The sum of allocation values for each worker type must be <=
        # the number of workers of that type.
        per_worker_type_allocations = []
        for i in range(m):
            relevant_vars = \
                x[:,i*num_vars_per_job:(i+1)*num_vars_per_job]
            relevant_scale_factors = \
                scale_factors_array[:,i*num_vars_per_job:(i+1)*num_vars_per_job]
            per_worker_type_allocations.append(
                cp.sum(cp.multiply(relevant_vars,
                                   cp.multiply(relevant_scale_factors,
                                               masks))))
        constraints.append(
            cp.atoms.affine.hstack.hstack(per_worker_type_allocations) <= \
                num_workers)

        # Set the following constraints:
        # for all job type pairs a, b:
        #   sum of allocation of all jobs of type a paired with type b ==
        #   sum of allocation of all jobs of type b paired with type a
        lhs = []
        rhs = []
        for i, job_type_0 in enumerate(job_types):
            for j, job_type_1 in enumerate(job_types):
                if j <= i:
                    continue

                # Retrieve the list of jobs of each type.
                job_type_0_jobs = job_type_to_job_idx[job_type_0]
                job_type_1_jobs = job_type_to_job_idx[job_type_1]

                for k in range(m):
                    job_type_0_mask = np.zeros(x.shape)
                    job_type_1_mask = np.zeros(x.shape)

                    # Allocation of job_type_0 jobs when paired with job_type_1
                    for job_idx in job_type_0_jobs:
                        offset = k * num_vars_per_job + 1 + j
                        job_type_0_mask[job_idx,offset] = 1

                    # Allocation of job_type_1 jobs when paired with job_type_0
                    for job_idx in job_type_1_jobs:
                        offset = k  * num_vars_per_job + 1 + i
                        job_type_1_mask[job_idx,offset] = 1

                    lhs.append(cp.sum(x[job_type_0_mask == 1]))
                    rhs.append(cp.sum(x[job_type_1_mask == 1]))

        constraints.append(cp.atoms.affine.hstack.hstack(lhs) ==
                           cp.atoms.affine.hstack.hstack(rhs))

        # Allocation coefficients.
        all_coefficients = np.zeros((n, num_vars_per_job * m))
        for i, job_id in enumerate(job_ids):
            job_type = job_id_to_job_type[job_id]
            job_type_idx = job_types.index(job_type)
            if len(job_type_to_job_idx[job_type]) == 1:
                for k, worker_type in enumerate(worker_types):
                    offset = k * num_vars_per_job + 1 + job_type_idx
                    constraints.append(x[i,offset] == 0.0)
            all_coefficients[i] = \
                np.multiply(flattened_job_type_throughputs[job_type_idx],
                            scale_factors_array[i]) / priority_weights[job_id]
        objective = \
            cp.Maximize(cp.min(cp.sum(cp.multiply(all_coefficients, x),
                                      axis=1)))

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver='ECOS')

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        allocation = x.value.clip(min=0.0).clip(max=1.0)

        return allocation


    def get_allocation_v2(self, unflattened_job_type_throughputs,
                          job_id_to_job_type, scale_factors,
                          priority_weights, cluster_spec):
        # TODO: Rename unflattened_job_type_throughputs ->
        #       unflattened_throughputs
        job_ids = sorted(job_id_to_job_type.keys())
        job_types = sorted(unflattened_job_type_throughputs.keys())
        worker_types = sorted(cluster_spec.keys())
        self._num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]

        # Create a map from job type to list of job indexes.
        job_type_to_job_idx = {}
        for i, job_id in enumerate(job_ids):
            job_type = job_id_to_job_type[job_id]
            if job_type not in job_type_to_job_idx:
                job_type_to_job_idx[job_type] = []
            job_type_to_job_idx[job_type].append(i)

        # Num jobs.
        n = len(job_ids)
        # Num job_types.
        a = len(unflattened_job_type_throughputs.keys())
        # Num worker_types.
        m = len(worker_types)
        # Num varibles per job.
        num_vars_per_job = 1 + a

        # Compute normalizing factor for each job type.
        # This normalizing factor will be used to normalize throughputs for the
        # same job type in job-job type combinations, as well.
        normalizing_factors = {}
        for job_type in job_types:
            unflattened_throughputs = \
                unflattened_job_type_throughputs[job_type]
            normalizing_factor = 0.0
            for worker_type in worker_types:
                normalizing_factor += \
                    unflattened_throughputs[worker_type][None]
            normalizing_factors[job_type] = normalizing_factor

        flattened_job_type_throughputs = np.zeros(shape=(a, 1 + a, m),
                                                  dtype=np.float32)
        for i, job_type in enumerate(job_types):
            unflattened_throughputs = \
                unflattened_job_type_throughputs[job_type]
            for j, other_job_type in enumerate([None] + job_types):
                for k, worker_type in enumerate(worker_types):
                    flattened_job_type_throughputs[i,j,k] = \
                        unflattened_throughputs[worker_type][other_job_type]
        for i, job_type in enumerate(job_types):
            flattened_job_type_throughputs[i] /= normalizing_factors[job_type]

        # Allocation matrix.
        x = cp.Variable((n * num_vars_per_job, m))

        # Set up masks to avoid double-counting allocation values when
        # computing constraint that the sum of allocation values of each
        # worker type must be <= the number of workers of that worker type.
        # TODO: Change this if we ever consider combinations larger than pairs.
        masks = np.ones((n * num_vars_per_job, m))
        for i in range(0, n * num_vars_per_job, num_vars_per_job):
            for j in range(1, num_vars_per_job):
                for k in range(m):
                    masks[i+j, k] = 0.5

        # Set up scale factors.
        scale_factors_array = np.ones((n * num_vars_per_job, m))
        for i in range(0, n * num_vars_per_job, num_vars_per_job):
            scale_factors_array[i:i+num_vars_per_job] = \
                scale_factors[job_ids[i // num_vars_per_job]]

        objective_terms = []
        constraints = [
            x >= 0,
        ]

        # Set the following constraints:
        # for all job type pairs a, b:
        #   sum of allocation of all jobs of type a paired with type b ==
        #   sum of allocation of all jobs of type b paired with type a
        for i, job_type_0 in enumerate(job_types):
            for j, job_type_1 in enumerate(job_types):
                if j <= i:
                    continue

                # Store the allocation values for jobs of each type
                job_type_0_job_allocations = []
                job_type_1_job_allocations = []

                # Retrieve the list of jobs of each type.
                job_type_0_jobs = job_type_to_job_idx[job_type_0]
                job_type_1_jobs = job_type_to_job_idx[job_type_1]

                # Allocation of jobs of job_type_0 when paired with job_type_1
                for job_idx in job_type_0_jobs:
                    job_id = job_ids[job_idx]
                    job_idx *= num_vars_per_job
                    job_type_0_job_allocations.append(x[job_idx+1+j])

                # Allocation of jobs of job_type_1 when paired with job_type_0
                for job_idx in job_type_1_jobs:
                    job_id = job_ids[job_idx]
                    job_idx *= num_vars_per_job
                    job_type_1_job_allocations.append(x[job_idx+1+i])

                constraints.append(cp.sum(job_type_0_job_allocations) ==
                                   cp.sum(job_type_1_job_allocations))

        for i in range(0, n * num_vars_per_job, num_vars_per_job):
            job_id = job_ids[i // num_vars_per_job]
            job_type = job_id_to_job_type[job_id]
            job_type_idx = job_types.index(job_type)
            # If there is only one job of this job type, zero out the
            # allocation corresponding to this job colocating with itself.
            if len(job_type_to_job_idx[job_type]) == 1:
                for k, worker_type in enumerate(worker_types):
                    constraints.append(x[i+1+job_type_idx,k] == 0.0)

            # Compute the effective throughput for each job.
            coefficients = \
                np.multiply(scale_factors_array[i:i+num_vars_per_job],
                            flattened_job_type_throughputs[job_type_idx] /\
                                priority_weights[job_id])
            objective_terms.append(
                cp.sum(cp.multiply(x[i:i+num_vars_per_job],
                                   coefficients)))
            constraints.append(cp.sum(x[i:i+num_vars_per_job]) <= 1)

        constraints.append(
            cp.sum(cp.multiply(x, cp.multiply(scale_factors_array, masks)),
                   axis=0) <= self._num_workers)

        if len(objective_terms) == 1:
            objective = cp.Maximize(objective_terms[0])
        else:
            objective = cp.Maximize(cp.minimum(*objective_terms))

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        allocation = x.value.clip(min=0.0).clip(max=1.0)

        return allocation


    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
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

        objective_terms = []
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs.
        for i in range(len(all_throughputs)):
            indexes = relevant_combinations[single_job_ids[i]]
            objective_terms.append(cp.sum(cp.multiply(
                np.multiply(all_throughputs[i][indexes],
                            scale_factors_array[indexes]), x[indexes])))
        if len(objective_terms) == 1:
            objective = cp.Maximize(objective_terms[0])
        else:
            objective = cp.Maximize(cp.minimum(*objective_terms))
        # Make sure the allocation can fit in the cluster.
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= self._num_workers,
        ]
        for single_job_id in single_job_ids:
            indexes = relevant_combinations[single_job_id]
            constraints.append(cp.sum(x[indexes]) <= 1)

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

class ThroughputNormalizedByCostSumWithPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'ThroughputNormalizedByCostSum_Perf'

    def get_allocation(self, unflattened_throughputs, cluster_spec,
                       instance_costs=None, SLAs={}, num_steps_remaining={}):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale = 1.0 / throughputs.sum(axis=1)
        throughputs = throughputs * scale.reshape(m, 1)

        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        if instance_costs is None:
            objective = cp.Maximize(cp.sum(cp.sum(cp.multiply(throughputs, x),
                                    axis=1)))
        else:
            instance_costs_array = np.zeros((1, n))
            for i in range(n):
                instance_costs_array[0, i] = instance_costs[worker_types[i]]
            objective = cp.Maximize(cp.sum(cp.sum(cp.multiply(throughputs / instance_costs_array, x),
                                    axis=1)))

        # Make sure that a given job is not over-allocated resources.
        # TODO: Account for self.num_workers and scale_factor_array?
        constraints = [
            x >= 0,
            cp.sum(x, axis=1) <= 1,
        ]
        for job_id in SLAs:
            i = job_ids.index(job_id)
            assert(job_id in num_steps_remaining)
            constraints.append(
                cp.sum(cp.multiply(throughputs / instance_costs_array, x),
                       axis=1) >= (num_steps_remaining[job_id] / SLAs[job_id])
            )
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)

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


class FIFOPolicy(Policy):
    def __init__(self, mode='base', seed=None, packing_threshold=1.5):
        self._name = 'FIFO'
        self._mode = mode
        self._allocation = {}
        self._scale_factors = {}
        if mode == 'base':
            self._rng = random.Random()
            if seed is not None:
                self._rng.seed(seed)
        elif mode == 'packing':
            self._packing_threshold = packing_threshold

    def _pack(self, queue, throughputs, scale_factors):
        while len(queue) > 0:
            # Only make a packing decision if combined normalized
            # throughput would provide a signficant gain.
            max_packed_throughput = self._packing_threshold
            job_id_to_pack_with = None
            job_id_to_schedule = queue.pop(0)

            # Find the already scheduled job with which the next job on
            # the queue will pack best with.
            for scheduled_job_id in self._allocation:
                assert scheduled_job_id != job_id_to_schedule
                assert scheduled_job_id in throughputs
                if scheduled_job_id.is_pair():
                    continue
                if (scale_factors[scheduled_job_id] !=\
                        scale_factors[job_id_to_schedule]):
                    continue
                worker_type = self._allocation[scheduled_job_id]
                merged_job_id = \
                        job_id_pair.JobIdPair(scheduled_job_id[0],
                                              job_id_to_schedule[0])
                packed_throughput = throughputs[merged_job_id][worker_type]
                normalized_packed_throughput = 0.0
                for i, single_job_id in enumerate(merged_job_id.singletons()):
                    if packed_throughput[i] <= 0.0:
                        continue
                    isolated_throughput = \
                            throughputs[single_job_id][worker_type]
                    normalized_packed_throughput += \
                            packed_throughput[i] / isolated_throughput
                if normalized_packed_throughput > max_packed_throughput:
                    max_packed_throughput = normalized_packed_throughput
                    job_id_to_pack_with = scheduled_job_id
            if job_id_to_pack_with is None:
                # Terminate when we cannot find a job to pack with.
                # This respects the FIFO property of no jobs being able
                # to jump ahead in the queue.
                break
            else:
                # Transfer the allocation for the single job to the
                # packed job.
                self._output = None
                merged_job_id = \
                        job_id_pair.JobIdPair(job_id_to_pack_with[0],
                                              job_id_to_schedule[0])
                worker_type = self._allocation[job_id_to_pack_with]
                del self._allocation[job_id_to_pack_with]
                self._allocation[merged_job_id] = worker_type


    def get_allocation(self, throughputs, scale_factors, cluster_spec):
        available_workers = copy.deepcopy(cluster_spec)
        queue = []

        # Update the internal representation of scale_factors.
        for job_id in scale_factors:
            self._scale_factors[job_id] = scale_factors[job_id]

        # Reset the allocation when running in performance-aware mode.
        if self._mode != 'base':
            self._allocation = {}

        # Add all jobs that have not been allocated already to the queue.
        # Jobs should be added in order of arrival (i.e. according to Job ID).
        for job_id in sorted(list(throughputs.keys())):
            if job_id not in self._allocation and not job_id.is_pair():
                queue.append(job_id)

        # Find all completed jobs and schedule jobs off the queue to replace
        # them. Also determine how many workers are available.
        # NOTE: In performance-aware mode, this loop should be a no-op
        # because the allocation is reset.
        for scheduled_job_id in sorted(list(self._allocation.keys())):
            worker_type = self._allocation[scheduled_job_id]
            # Check if job has completed.
            if scheduled_job_id not in throughputs:
                # If only one job in a pair of co-located jobs completed, then
                # add the other job back to the queue.
                for single_job_id in scheduled_job_id.singletons():
                    if single_job_id in throughputs:
                        queue.append(single_job_id)
                        queue.sort()
                if len(queue) > 0:
                    job_id_to_schedule = queue.pop(0)
                    if (scale_factors[job_id_to_schedule] <=
                            available_workers[worker_type]):
                        worker_type = self._allocation[scheduled_job_id]
                        self._allocation[job_id_to_schedule] = worker_type
                        available_workers[worker_type] -= \
                            scale_factors[job_id_to_schedule]
                del self._allocation[scheduled_job_id]
                del self._scale_factors[scheduled_job_id]
            else:
                # Job has not completed, subtract its allocated workers
                # from available_workers.
                available_workers[worker_type] -= \
                    scale_factors[scheduled_job_id]

        # Find all available workers.
        available_worker_types = []
        for worker_type in available_workers:
            if available_workers[worker_type] > 0:
                available_worker_types.append(worker_type)
        available_worker_types.sort()

        # Allocate resources to as many jobs as possible.
        while len(queue) > 0 and len(available_worker_types) > 0:
            job_id_to_schedule = queue.pop(0)
            scale_factor = scale_factors[job_id_to_schedule]
            available_worker_types_with_scale_factor = []
            original_available_worker_types_mapping = []
            for i, worker_type in enumerate(available_worker_types):
                if available_workers[worker_type] >= scale_factor:
                    available_worker_types_with_scale_factor.append(worker_type)
                    original_available_worker_types_mapping.append(i)
            if len(available_worker_types_with_scale_factor) == 0:
                break
            if self._mode == 'base':
                worker_type_idx = self._rng.randrange(
                        len(available_worker_types_with_scale_factor))
            else:
                # Find the worker_type with best performance for this job.
                worker_type = None
                worker_type_idx = None
                max_throughput = -1
                for i, x in enumerate(available_worker_types_with_scale_factor):
                    throughput = throughputs[job_id_to_schedule][x]
                    if throughput > max_throughput:
                        max_throughput = throughput
                        worker_type = x
                        worker_type_idx = i
            self._allocation[job_id_to_schedule] = worker_type
            available_workers[worker_type] -= scale_factors[job_id_to_schedule]
            if available_workers[worker_type] == 0:
                worker_type_idx =\
                    original_available_worker_types_mapping[worker_type_idx]
                available_worker_types.pop(worker_type_idx)

        if self._mode == 'packing':
            self._pack(queue, throughputs, scale_factors)

        # Construct output allocation.
        final_allocation = {}
        for job_id in throughputs:
            final_allocation[job_id] = \
                    {worker_type: 0.0 for worker_type in cluster_spec}
        for job_id, worker_type in self._allocation.items():
            final_allocation[job_id][worker_type] = 1.0

        return final_allocation

class FIFOPolicyWithPerf(Policy):
    def __init__(self, packing=False):
        self._name = 'FIFO_Perf'
        self._packing = packing
        self._policy = FIFOPolicy(mode='perf')

    def get_allocation(self, throughputs, scale_factors, cluster_spec):
        return self._policy.get_allocation(throughputs, scale_factors,
                                           cluster_spec)

class FIFOPolicyWithPacking(PolicyWithPacking):
    def __init__(self, packing_threshold=1.5):
        self._name = 'FIFO_Packing'
        self._policy = FIFOPolicy(mode='packing',
                                  packing_threshold=packing_threshold)

    def get_allocation(self, throughputs, scale_factors, cluster_spec):
        return self._policy.get_allocation(throughputs, scale_factors,
                                           cluster_spec)
