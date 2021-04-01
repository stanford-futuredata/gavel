import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import cvxpy as cp
import numpy as np

import job_id_pair

class Policy:

    def __init__(self, solver='ECOS'):
        self._name = None
        self._solver = solver

    @property
    def name(self):
        return self._name

    def scale_factors_array(self, scale_factors, job_ids, m, n):
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]
        return scale_factors_array

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

    def get_base_constraints(self, x, scale_factors_array):
        """Return base constraints."""
        return [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= self._num_workers,
            cp.sum(x, axis=1) <= 1,
        ]


class PolicyWithPacking(Policy):

    def __init__(self, solver='ECOS'):
        Policy.__init__(self, solver)

    def scale_factors_array(self, scale_factors, job_ids, m, n):
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
        return scale_factors_array

    def flatten(self, d, cluster_spec, priority_weights=None):
        """
        Converts a 2-level dict to a NumPy array.

        Job ID combinations in the input dict are either a tuple or an integer.
        If a tuple, represents a combination run on a GPU concurrently.
        If an integer, represents a single job / application run on the
        GPU.

        Returns a list of each user's throughput matrix and an
        index to reconstruct the allocation as a dict.
        """
        job_ids = sorted(list(d.keys()))
        if len(job_ids) == 0:
            return None, None
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

        if len(worker_types) == 0:
            return None, None

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

    def get_base_constraints(self, x, single_job_ids,
                             scale_factors_array, relevant_combinations):
        """Return base constraints."""
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= np.array(self._num_workers),
        ]

        # Every job cannot receive a total time share sum greater than 1.0.
        idx = []
        for single_job_id in single_job_ids:
            indexes = relevant_combinations[single_job_id]
            idx += indexes
        index_var = x[idx]
        index_var = cp.reshape(index_var,
            (len(single_job_ids), int(np.prod(index_var.shape) /
             len(single_job_ids))), order='C')
        constraints.append(cp.sum(index_var, axis=1) <= 1)
        return constraints

    def convert_job_type_allocation(self, allocation, job_id_to_job_type_key):
        """Converts a job-job_type allocation to a job-job allocation."""
        job_ids = sorted(allocation.keys())
        worker_types = sorted(allocation[job_ids[0]].keys())
        job_type_keys = \
            sorted(set([job_id_to_job_type_key[job_id] for job_id in job_ids]))

        # Initialize job_type-job_type allocation.
        job_type_allocation = {}
        for worker_type in worker_types:
            job_type_allocation[worker_type] = {}
            for job_type_key in job_type_keys:
                job_type_allocation[worker_type][job_type_key] = {}
                job_type_allocation_ = \
                    job_type_allocation[worker_type][job_type_key]
                for other_job_type_key in [None] + job_type_keys:
                    job_type_allocation_[other_job_type_key] = 0.0

        # Populate job_type-job_type allocation.
        for worker_type in worker_types:
            for job_id in allocation:
                job_type_key = job_id_to_job_type_key[job_id]
                for other_job_type_key in allocation[job_id][worker_type]:
                    job_type_allocation[worker_type][job_type_key][other_job_type_key] += \
                        allocation[job_id][worker_type][other_job_type_key]

        # Compute job-job allocations using the following formula:
        # x_{i,j} = x_{i, job_type(j)} * x_{j, job_type(i)} /
        #   sum x_{k, job_type(j)} for all k of job_type(i)
        converted_allocation = {}
        for i, job_id in enumerate(job_ids):
            converted_allocation[job_id] = {}
            job_type_key = job_id_to_job_type_key[job_id]
            # Set the isolated allocations.
            for worker_type in worker_types:
                converted_allocation[job_id][worker_type] = \
                    allocation[job_id][worker_type][None]
            # Set the packed allocations.
            for other_job_id in job_ids[i+1:]:
                other_job_type_key = job_id_to_job_type_key[other_job_id]
                merged_job_id = \
                    job_id_pair.JobIdPair(job_id[0], other_job_id[0])
                converted_allocation[merged_job_id] = {}
                for worker_type in worker_types:
                    current_job_type_allocation = \
                        job_type_allocation[worker_type][job_type_key][other_job_type_key]
                    if current_job_type_allocation > 0.0:
                        if job_type_key == other_job_type_key:
                            current_job_type_allocation -= \
                                allocation[job_id][worker_type][job_type_key]
                        converted_allocation[merged_job_id][worker_type] = \
                            (allocation[job_id][worker_type][other_job_type_key] *\
                             allocation[other_job_id][worker_type][job_type_key] /\
                             current_job_type_allocation)
                    else:
                        converted_allocation[merged_job_id][worker_type] = 0.0

        return converted_allocation
