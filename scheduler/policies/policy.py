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
                scale_factors_array, x), axis=0) <= self._num_workers,
        ]

        # Every job cannot receive a total time share sum greater than 1.0.
        per_job_allocations = []
        for single_job_id in single_job_ids:
            indexes = relevant_combinations[single_job_id]
            per_job_allocations.append(cp.sum(x[indexes]))
        constraints.append(cp.vstack(per_job_allocations) <= 1)

        return constraints

    def convert_job_type_allocation(self, allocation, job_id_to_job_type):
        """Converts a job-job_type allocation to a job-job allocation."""
        job_ids = sorted(allocation.keys())
        worker_types = sorted(allocation[job_ids[0]].keys())
        job_types = \
            sorted(set([job_id_to_job_type[job_id] for job_id in job_ids]))

        # Initialize job_type-job_type allocation.
        job_type_allocation = {}
        for worker_type in worker_types:
            job_type_allocation[worker_type] = {}
            for job_type in job_types:
                job_type_allocation[worker_type][job_type] = {}
                job_type_allocation_ = \
                    job_type_allocation[worker_type][job_type]
                for other_job_type in [None] + job_types:
                    job_type_allocation_[other_job_type] = 0.0

        # Populate job_type-job_type allocation.
        for worker_type in worker_types:
            for job_id in allocation:
                job_type = job_id_to_job_type[job_id]
                for other_job_type in allocation[job_id][worker_type]:
                    job_type_allocation[worker_type][job_type][other_job_type] += \
                        allocation[job_id][worker_type][other_job_type]

        # Compute job-job allocations using the following formula:
        # x_{i,j} = x_{i, job_type(j)} * x_{j, job_type(i)} /
        #   sum x_{k, job_type(j)} for all k of job_type(i)
        converted_allocation = {}
        for i, job_id in enumerate(job_ids):
            converted_allocation[job_id] = {}
            job_type = job_id_to_job_type[job_id]
            # Set the isolated allocations.
            for worker_type in worker_types:
                converted_allocation[job_id][worker_type] = \
                    allocation[job_id][worker_type][None]
            # Set the packed allocations.
            for other_job_id in job_ids[i+1:]:
                other_job_type = job_id_to_job_type[other_job_id]
                merged_job_id = \
                    job_id_pair.JobIdPair(job_id[0], other_job_id[0])
                converted_allocation[merged_job_id] = {}
                for worker_type in worker_types:
                    current_job_type_allocation = \
                        job_type_allocation[worker_type][job_type][other_job_type]
                    if current_job_type_allocation > 0.0:
                        if job_type == other_job_type:
                            current_job_type_allocation -= \
                                allocation[job_id][worker_type][job_type]
                        converted_allocation[merged_job_id][worker_type] = \
                            (allocation[job_id][worker_type][other_job_type] *\
                             allocation[other_job_id][worker_type][job_type] /\
                             current_job_type_allocation)
                    else:
                        converted_allocation[merged_job_id][worker_type] = 0.0

        return converted_allocation
