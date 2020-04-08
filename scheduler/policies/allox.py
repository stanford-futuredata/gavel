import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

from policy import Policy, PolicyWithPacking

class AlloXPolicy(Policy):
    def __init__(self):
        self._name = 'AlloX'
        self._prev_allocation = {}

    def get_allocation(self, unflattened_throughputs,
                       scale_factors, num_steps_remaining,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        # Make sure all scale factors are 1, since AlloX only supports jobs
        # with scale factors of 1.
        for job_id in scale_factors:
            assert(scale_factors[job_id] == 1)

        # m is the number of jobs, n is the total number of available workers
        # (not the total number of worker types).
        unallocated_job_ids = []
        already_allocated_job_ids = []
        for job_id in unflattened_throughputs:
            if job_id not in self._prev_allocation:
                unallocated_job_ids.append(job_id)
            else:
                already_allocated_job_ids.append(job_id)
        m = len(unallocated_job_ids)

        n = 0
        worker_id_to_worker_type_mapping = {}
        for worker_type in worker_types:
            num_workers = cluster_spec[worker_type]
            for already_allocated_job_id in already_allocated_job_ids:
                if self._prev_allocation[already_allocated_job_id][worker_type] == 1.0:
                    num_workers -= 1
            for worker_id in range(n, n+num_workers):
                worker_id_to_worker_type_mapping[worker_id] = worker_type
                n += 1

        # Construct matrix of processing times for each job on each worker,
        # taking into account the type of each worker.
        q_base = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                worker_type = worker_id_to_worker_type_mapping[j]
                q_base[i, j] = num_steps_remaining[unallocated_job_ids[i]] / \
                    unflattened_throughputs[unallocated_job_ids[i]][worker_type]
        # q is computed as [q_base q_base*2 q_base*3 ... q_base*n].
        q = np.copy(q_base)
        for i in range(2, m+1):
            scaled_q_base = i * q_base
            q = np.concatenate((q, scaled_q_base), axis=1)

        # Solve assignment problem using Hungarian method (implemented in scipy).
        row_indices, col_indices = linear_sum_assignment(q)

        # Extract assignment of jobs to worker types.
        per_worker_id_job_assignment = {i: [] for i in range(n)}
        for (row_index, col_index) in zip(row_indices, col_indices):
            job_id = unallocated_job_ids[row_index]
            worker_id = col_index % n
            worker_type = worker_id_to_worker_type_mapping[worker_id]
            worker_type_order = col_index // n
            per_worker_id_job_assignment[worker_id].append((job_id, worker_type_order))
        for worker_id in range(n):
            per_worker_id_job_assignment[worker_id] = [
                (x[0], len(per_worker_id_job_assignment[worker_id]) -1 - x[1])
                for x in per_worker_id_job_assignment[worker_id]]
            per_worker_id_job_assignment[worker_id].sort(key=lambda x: x[1])

        # Construct allocation. Don't remember allocations beyond the first
        # for each worker, since these can be recomputed the next time the
        # policy is run. Copy over allocations from already running jobs whose
        # allocations have already been computed.
        allocation = {}
        for job_id in job_ids:
            allocation[job_id] = \
                {worker_type: 0.0 for worker_type in cluster_spec}
        for worker_id in range(n):
            if len(per_worker_id_job_assignment[worker_id]) > 0:
                job_id = per_worker_id_job_assignment[worker_id][0][0]
                worker_type = worker_id_to_worker_type_mapping[worker_id]
                allocation[job_id][worker_type] = 1.0
        for job_id in job_ids:
            if job_id in self._prev_allocation:
                allocation[job_id] = copy.copy(self._prev_allocation[job_id])
        self._prev_allocation = copy.copy(allocation)

        return allocation
