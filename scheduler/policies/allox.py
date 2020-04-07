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

        # TODO: Don't change the allocation for unfinished jobs (AlloX does not
        # support preemption).
        # TODO: Support online arrival of jobs.

        # m is the number of jobs, n is the total number of workers (not the
        # total number of worker types).
        (m, _) = throughputs.shape
        n = 0
        worker_id_to_worker_type_mapping = {}
        for worker_type in worker_types:
            for worker_id in range(n, n+cluster_spec[worker_type]):
                worker_id_to_worker_type_mapping[worker_id] = worker_type
            n += cluster_spec[worker_type]

        # Construct matrix of processing times for each job on each worker.
        q_base = np.zeros((m, n))
        for i in range(m):
            j_counter = 0
            for worker_type in worker_types:
                for j in range(j_counter, j_counter+cluster_spec[worker_type]):
                    q_base[i, j] = num_steps_remaining[job_ids[i]] / \
                        unflattened_throughputs[job_ids[i]][worker_type]
                j_counter += cluster_spec[worker_type]
        # q = [q_base q_base*2 q_base*3 ... q_base*n].
        q = np.copy(q_base)
        for i in range(2, m+1):
            scaled_q_base = i * q_base
            q = np.concatenate((q, scaled_q_base), axis=1)

        # Solve assignment problem using Hungarian method (implemented in scipy).
        row_indices, col_indices = linear_sum_assignment(q)

        # Extract assignment of jobs to worker types.
        worker_order = {i: [] for i in range(n)}
        for (row_index, col_index) in zip(row_indices, col_indices):
            job_id = row_index
            worker_id = col_index % n
            worker_type = worker_id_to_worker_type_mapping[worker_id]
            worker_type_order = col_index // n
            worker_order[worker_id].append((job_id, worker_type_order))
        for worker_id in range(n):
            worker_order[worker_id] = [(x[0], len(worker_order[worker_id]) -1 - x[1])
                                       for x in worker_order[worker_id]]
            worker_order[worker_id].sort(key=lambda x: x[1])
            worker_type = worker_id_to_worker_type_mapping[worker_id]
            print(worker_id, worker_id_to_worker_type_mapping[worker_id],
                  [x[0] for x in worker_order[worker_id]],
                  [num_steps_remaining[job_ids[x[0]]] / unflattened_throughputs[job_ids[x[0]]][worker_type]
                   for x in worker_order[worker_id]])

        # Construct allocation. Don't remember allocations beyond the first
        # for each worker, since these can be recomputed the next time the
        # policy is run.
        allocation = {}
        for job_id in job_ids:
            allocation[job_id] = \
                {worker_type: 0.0 for worker_type in cluster_spec}
        for worker_id in range(n):
            job_id = worker_order[worker_id][0][0]
            worker_type = worker_id_to_worker_type_mapping[worker_id]
            allocation[job_id][worker_type] = 1.0
        self._prev_allocation = copy.copy(allocation)

        return allocation
