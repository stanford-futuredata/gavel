import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy

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
        self._proportional_policy = ProportionalPolicy()

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
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

        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.min(cp.sum(cp.multiply(
                np.multiply(throughputs * priority_weights.reshape((m, 1)),
                            scale_factors_array), x), axis=1)))
        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


class MaxMinFairnessPolicyWithPacking(PolicyWithPacking):

    def __init__(self, solver):
        PolicyWithPacking.__init__(self, solver)
        self._name = 'MaxMinFairness_Packing'
        self._proportional_policy = ProportionalPolicy()

    def get_allocation_using_job_type_throughputs(
            self, unflattened_throughputs, job_id_to_job_type_key,
            scale_factors, unflattened_priority_weights, cluster_spec):
        job_ids = sorted(job_id_to_job_type_key.keys())
        if len(job_ids) == 0:
            return None
        job_type_keys = sorted(unflattened_throughputs.keys())
        worker_types = sorted(cluster_spec.keys())
        num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]

        # Create a map from job type to list of job indexes.
        job_type_key_to_job_idx = {}
        for i, job_id in enumerate(job_ids):
            job_type_key = job_id_to_job_type_key[job_id]
            if job_type_key not in job_type_key_to_job_idx:
                job_type_key_to_job_idx[job_type_key] = []
            job_type_key_to_job_idx[job_type_key].append(i)

        # Num jobs.
        n = len(job_ids)
        # Num job_types.
        a = len(unflattened_throughputs.keys())
        # Num worker_types.
        m = len(worker_types)
        # Num varibles per job.
        num_vars_per_job = 1 + a

        # Set up scale factors.
        flattened_scale_factors = \
            np.reshape([scale_factors[job_id] for job_id in job_ids], (n, 1))
        scale_factors_array = np.tile(flattened_scale_factors,
                                        (1, num_vars_per_job * m))

        # Set up flattened job type throughputs.
        flattened_throughputs = np.zeros(shape=(a, (1 + a) * m),
                                         dtype=np.float32)
        for i, job_type_key in enumerate(job_type_keys):
            for k, worker_type in enumerate(worker_types):
                for j, other_job_type_key in enumerate([None] + job_type_keys):
                    if j > 0 and other_job_type_key[1] != job_type_key[1]:
                        flattened_throughputs[i,k*(1+a)+j] = 0.0
                    else:
                        flattened_throughputs[i,k*(1+a)+j] = \
                            unflattened_throughputs[job_type_key][worker_type][other_job_type_key]

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
                cp.hstack(per_worker_type_allocations) <= num_workers)

        # Set the following constraints:
        # for all job type pairs a, b:
        #   sum of allocation of all jobs of type a paired with type b ==
        #   sum of allocation of all jobs of type b paired with type a
        lhs = []
        rhs = []
        for i, job_type_key_0 in enumerate(job_type_keys):
            for j, job_type_key_1 in enumerate(job_type_keys):
                if j <= i:
                    continue
                elif job_type_key_0[1] != job_type_key_1[1]:
                    continue

                # Retrieve the list of jobs of each type.
                job_type_0_jobs = job_type_key_to_job_idx[job_type_key_0]
                job_type_1_jobs = job_type_key_to_job_idx[job_type_key_1]

                for k in range(m):
                    job_type_0_mask = np.zeros(x.shape)
                    job_type_1_mask = np.zeros(x.shape)

                    # Allocation of job_type_0 jobs when paired with job_type_1
                    for job_idx in job_type_0_jobs:
                        offset = k * num_vars_per_job + 1 + j
                        job_type_0_mask[job_idx,offset] = 1

                    # Allocation of job_type_1 jobs when paired with job_type_0
                    for job_idx in job_type_1_jobs:
                        offset = k * num_vars_per_job + 1 + i
                        job_type_1_mask[job_idx,offset] = 1

                    lhs.append(cp.sum(x[job_type_0_mask == 1]))
                    rhs.append(cp.sum(x[job_type_1_mask == 1]))

        assert (len(lhs) == len(rhs))
        if len(lhs) > 0:
            constraints.append(cp.hstack(lhs) == cp.hstack(rhs))

        # Add constraints to make all variables of the form i-A where job i
        # is of job type A equal.
        for i, job_type_key in enumerate(job_type_keys):
            for k in range(m):
                same_job_type_vars = []
                job_type_jobs = job_type_key_to_job_idx[job_type_key]

                # Find all variables for job-job_type pairs where the job
                # types match.
                offset = k * num_vars_per_job + 1 + i
                for job_idx in job_type_jobs:
                    same_job_type_vars.append(x[job_idx, offset])

                # Constrain the variables to all be equal.
                c = cp.Variable()
                constraints.append(cp.hstack(same_job_type_vars) == c)

        throughputs_no_packed_jobs = np.zeros((len(job_ids), len(worker_types)))
        for i, job_id in enumerate(job_ids):
            job_type_key = job_id_to_job_type_key[job_id]
            for j, worker_type in enumerate(worker_types):
                throughputs_no_packed_jobs[i, j] = \
                    unflattened_throughputs[job_type_key][worker_type][None]
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs_no_packed_jobs,
            (job_ids, worker_types),
            cluster_spec)

        # Allocation coefficients.
        all_coefficients = np.zeros((n, num_vars_per_job * m))
        for i, job_id in enumerate(job_ids):
            job_type_key = job_id_to_job_type_key[job_id]
            job_type_idx = job_type_keys.index(job_type_key)
            if len(job_type_key_to_job_idx[job_type_key]) == 1:
                for k, worker_type in enumerate(worker_types):
                    offset = k * num_vars_per_job + 1 + job_type_idx
                    constraints.append(x[i,offset] == 0.0)
            proportional_throughput = proportional_throughputs[i]
            all_coefficients[i] = \
                np.multiply(flattened_throughputs[job_type_idx],
                            scale_factors_array[i]) /\
                    (unflattened_priority_weights[job_id] * proportional_throughput)
        objective = \
            cp.Maximize(cp.min(cp.sum(cp.multiply(all_coefficients, x),
                                      axis=1)))

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        allocation = x.value.clip(min=0.0).clip(max=1.0)

        # Unflatten allocation.
        unflattened_allocation = {}
        for i, job_id in enumerate(job_ids):
            unflattened_allocation[job_id] = {}
            for j, worker_type in enumerate(worker_types):
                unflattened_allocation[job_id][worker_type] = {}
                for k, job_type_key in enumerate([None] + job_type_keys):
                    unflattened_allocation[job_id][worker_type][job_type_key] = \
                        allocation[i, j * num_vars_per_job + k]

        return self.convert_job_type_allocation(unflattened_allocation,
                                                job_id_to_job_type_key)

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

        objective_terms = []
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs.
        import scipy.sparse as sp
        idx = []
        tputs = []
        # compute the obejctive in a vectorized fashion
        for i in range(len(all_throughputs)):
            indexes = relevant_combinations[single_job_ids[i]]
            idx += indexes
            proportional_throughput = float(proportional_throughputs[i])
            curr_throughputs = np.multiply(
                    all_throughputs[i][indexes],
                    scale_factors_array[indexes]) / proportional_throughput
            tputs.append(curr_throughputs)

        tputs = sp.csc_matrix(np.vstack(tputs))
        indexed_vars = x[idx]
        realized_tputs = cp.multiply(tputs, indexed_vars)
        # reshape so that the sum of each row gives the throughput
        realized_tputs_mat = cp.reshape(realized_tputs,
                (len(all_throughputs),
                int(np.prod(realized_tputs.shape) / len(all_throughputs))),
                order='C')

        objective_fn = cp.min(cp.sum(realized_tputs_mat, axis=1))

        objective = cp.Maximize(objective_fn)

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
        if self._solver == 'SCS':
            # anderson acceleration is sometimes unstable, and adds
            # significant overhead
            kwargs = {'acceleration_lookback': 0}
        else:
            kwargs = {}

        result = cvxprob.solve(solver=self._solver, **kwargs)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return self.unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
