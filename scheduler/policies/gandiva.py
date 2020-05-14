import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import random

import job_id_pair
from policy import PolicyWithPacking

class GandivaPolicy(PolicyWithPacking):

    def __init__(self, seed=None):
        self._name = 'Gandiva_Packing'
        self._assigned_combinations = {}
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)

    def _get_allocation(self, job_combinations_to_schedule, index, scale_factors,
                        cluster_spec):
        # Helper method that divides time equally among all job combinations in
        # job_combinations_to_schedule.

        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        m = len(job_combinations_to_schedule)

        job_combination_indices_to_schedule = []
        for job_combination_to_schedule in job_combinations_to_schedule:
            job_combination_indices_to_schedule.append(
                job_ids.index(job_combination_to_schedule))

        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, len(job_ids), len(worker_types))

        # Split cluster over users (m). By construction,
        # \sum_i (x[i, j] * scale_factor[i]) = num_workers[j].
        # Normalize to ensure \sum_j x[i, j] <= 1 for all i.
        x = np.zeros((len(job_ids), len(worker_types)))
        for i in job_combination_indices_to_schedule:
            x[i] = np.array([cluster_spec[worker_type] / m for worker_type in worker_types])
            x[i] = x[i] / scale_factors_array[i]
        per_row_sum = np.sum(x, axis=1)
        per_row_sum = np.maximum(per_row_sum, np.ones(per_row_sum.shape))
        x = x / per_row_sum[:, None]

        return x

    def _get_normalized_throughput(self, job_combination, throughputs, worker_types):
        normalized_packed_throughput = 0.0
        if not job_combination.is_pair():
            return 0.0
        for worker_type in worker_types:
            packed_throughput = throughputs[job_combination][worker_type]
            for i, single_job_id in enumerate(job_combination.singletons()):
                if packed_throughput[i] <= 0.0:
                    return 0.0
                isolated_throughput = \
                    throughputs[single_job_id][worker_type]
                normalized_packed_throughput += \
                    packed_throughput[i] / isolated_throughput
        return normalized_packed_throughput

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec):
        all_throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None

        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index

        assigned_combination_keys = self._assigned_combinations.keys()
        to_delete = []
        for job_id in assigned_combination_keys:
            (job_combination, other_job_id) = self._assigned_combinations[job_id]
            if job_id not in job_ids:
                to_delete.extend([job_id, other_job_id])
                continue
            if other_job_id is not None:
                if other_job_id not in job_ids:
                    to_delete.extend([job_id, other_job_id])
                    continue
            # Stop using combinations with normalized throughput < 1.0.
            if self._get_normalized_throughput(job_combination, unflattened_throughputs,
                                               worker_types) < 1.0:
                to_delete.extend([job_id, other_job_id])
        for job_id in to_delete:
            if job_id is not None:
                if job_id in self._assigned_combinations:
                    del self._assigned_combinations[job_id]

        num_workers_requested = 0
        for single_job_id in single_job_ids:
            num_workers_requested += scale_factors[single_job_id]
        num_workers_available = 0
        for worker_type in worker_types:
            num_workers_available += cluster_spec[worker_type]

        if num_workers_requested <= num_workers_available:
            # If jobs fit in cluster, do not deploy packing.
            x = self._get_allocation(single_job_ids, index, scale_factors,
                                     cluster_spec)
        else:
            # Deploy packing.
            # Assign all job IDs that are not yet in combinations to combinations.
            to_be_assigned_combinations = []
            for single_job_id in single_job_ids:
                if single_job_id not in self._assigned_combinations:
                    to_be_assigned_combinations.append(single_job_id)

            # Randomly group jobs.
            i = 0
            n = len(to_be_assigned_combinations)
            while len(to_be_assigned_combinations) > 1 and i < n:
                i += 1
                [job1_id, job2_id] = self._rng.sample(
                    to_be_assigned_combinations, 2)
                # Make sure scale factors of jobs in job combination are the
                # same.
                if scale_factors[job1_id] != scale_factors[job2_id]:
                    continue
                to_be_assigned_combinations.remove(job1_id)
                to_be_assigned_combinations.remove(job2_id)
                job_combination = job_id_pair.JobIdPair(job1_id[0], job2_id[0])
                self._assigned_combinations[job1_id] = (job_combination,
                                                        job2_id)
                self._assigned_combinations[job2_id] = (job_combination,
                                                        job1_id)
            for i in range(len(to_be_assigned_combinations)):
                job_id = to_be_assigned_combinations[i]
                self._assigned_combinations[job_id] = (job_id, None)

            job_combinations_to_schedule = set()
            for single_job_id in self._assigned_combinations:
                job_combinations_to_schedule.add(
                    self._assigned_combinations[single_job_id][0])
            job_combinations_to_schedule = list(job_combinations_to_schedule)

            x = self._get_allocation(job_combinations_to_schedule, index,
                                     scale_factors,
                                     cluster_spec)

        return super().unflatten(x, index)
