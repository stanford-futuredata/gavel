import cvxpy as cp
import numpy as np


class Policy:
    def flatten(self, d):
        """Converts a 2-level dict to a NumPy array."""

        job_ids = list(d.keys())
        if len(job_ids) == 0:
            return None, None
        worker_types = list(d[job_ids[0]].keys())
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


class IsolatedPolicy(Policy):

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        return super().unflatten(np.full((m, n), 1.0 / (m * n)), index)


class MaximumThroughputPolicy(Policy):

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(cp.sum(cp.sum(cp.multiply(throughputs, x), axis=1)))
        constraints = [
            x >= 0,
            cp.sum(x, axis=0) <= 1,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        assert cvxprob.status == "optimal"
        return super().unflatten(x.value.clip(min=0.0), index)


class KSPolicy(Policy):

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        scale = 1.0 / throughputs.sum(axis=1)
        throughputs = throughputs * scale.reshape(m, 1)

        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(cp.min(cp.sum(cp.multiply(throughputs, x), axis=1)))
        constraints = [
            x >= 0,
            cp.sum(x, axis=0) <= 1,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        assert cvxprob.status == "optimal"

        return super().unflatten(x.value.clip(min=0.0), index)


class KSPolicyWithPacking(Policy):

    def flatten(self, d):
        """
        Converts a 2-level dict to a NumPy array.

        Job ID combinations in the input dict are either a tuple or an integer.
        If a tuple, represents a combination run on a GPU concurrently.
        If an integer, represents a single job / application run on the
        GPU.

        Returns a list of each user's normalized throughput matrix, a list
        of masks (used for normalization in the linear program), and an
        index to reconstruct the allocation as a dict.
        """

        job_ids = list(d.keys())
        if len(job_ids) == 0:
            return None, None, None
        worker_types = list(d[job_ids[0]].keys())
        individual_job_ids = []
        for job_id in job_ids:
            for individual_job_id in job_id.singletons():
                individual_job_ids.append(individual_job_id)

        # Compute normalizing factor for each individual job, this normalizing
        # factor will be used to normalize throughputs for the same job in job
        # combinations as well.
        normalizing_factors = {}
        for individual_job_id in individual_job_ids:
            normalizing_factor = 0.0
            for worker_type in worker_types:
                normalizing_factor += d[individual_job_id][worker_type]
            normalizing_factors[individual_job_id] = normalizing_factor

        if len(worker_types) == 0:
            return None, None, None

        all_m = []
        masks = []
        # Compute the throughput matrix and mask for each individual job.
        for individual_job_id in individual_job_ids:
            m = []
            mask = []
            # Each throughput matrix and mask has dimension
            # (num_app_combinations x num_worker_types).
            for job_id in job_ids:
                m_row = []
                mask_row = []
                for worker_type in worker_types:
                    # If job ID of interest is not in this job_id_combination,
                    # mask and throughput should be 0.
                    # Otherwise, use the right throughput from the input dict.
                    if job_id in individual_job_ids:
                        if job_id != individual_job_id:
                            m_row.append(0.0)
                            mask_row.append(0.0)
                        else:
                            m_row.append(d[job_id][worker_type])
                            mask_row.append(1.0)
                    else:
                        if not individual_job_id.overlaps_with(job_id):
                            m_row.append(0.0)
                            mask_row.append(1.0)
                        else:
                            # Find the index of the job of interest in the job
                            # combination tuple.
                            index = job_id.as_tuple().index(individual_job_id[0])
                            throughputs = d[job_id][worker_type]
                            m_row.append(d[job_id][worker_type][index])
                            mask_row.append(1.0)
                m.append(m_row)
                mask.append(mask_row)
            all_m.append(np.array(m) / normalizing_factors[individual_job_id])  # Normalize.
            masks.append(np.array(mask))
        return all_m, masks, (job_ids, individual_job_ids, worker_types)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_id_combinations, individual_job_ids, worker_types) = index
        d = {}
        for i in range(len(job_id_combinations)):
            d[job_id_combinations[i]] = {}
            for j in range(len(worker_types)):
                d[job_id_combinations[i]][worker_types[j]] = m[i][j]
        return d

    def get_allocation(self, unflattened_throughputs):
        all_throughputs, masks, index = self.flatten(unflattened_throughputs)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        x = cp.Variable(all_throughputs[0].shape)
        objective_terms = []
        for throughputs in all_throughputs:
            objective_terms.append(cp.sum(cp.multiply(throughputs, x)))
        if len(objective_terms) == 1:
            objective = cp.Maximize(objective_terms[0])
        else:
            objective = cp.Maximize(cp.minimum(*objective_terms))
        constraints = [
            x >= 0,
            cp.sum(x, axis=0) <= 1,  # One of these is redundant.
            cp.sum(x, axis=1) <= 1,
        ]
        for mask in masks:
            constraints.append(cp.sum(cp.multiply(x, mask)) <= 1)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        x = x.value.clip(min=0.0)
        x[x < 1e-2] = 0.0
        return self.unflatten(x,
                              index)

class FIFOPolicy(Policy):
    def __init__(self):
        self._allocation = {}
        self._queue = []

    def get_allocation(self, throughputs):
        # New Job ID; put on queue to schedule.
        job_id = None
        for job_id in throughputs:
            if job_id not in self._allocation and job_id not in self._queue:
                self._queue.append(job_id)

        # Old Job ID that has been removed; schedule job from queue.
        job_ids = list(self._allocation.keys())
        for job_id in job_ids:
            if job_id not in throughputs:
                worker_id = self._allocation[job_id]
                del self._allocation[job_id]
                if len(self._queue) > 0:
                    job_id_to_schedule = self._queue.pop(0)
                    self._allocation[job_id_to_schedule] = worker_id

        # worker_ids_seen keeps track of all workers that have been assigned
        # jobs.
        worker_ids_seen = set()
        for job_id in self._allocation:
            worker_id = self._allocation[job_id]
            worker_ids_seen.add(worker_id)

        # Try to allocation all queued job IDs on available workers.
        job_ids = list(throughputs.keys())
        if len(job_ids) > 0:
            job_id = job_ids[0]
            for worker_id in throughputs[job_id]:
                if worker_id not in worker_ids_seen:
                    if len(self._queue) > 0:
                        job_id_to_schedule = self._queue.pop(0)
                        self._allocation[job_id_to_schedule] = worker_id

        # Construct output allocation.
        allocation = {}
        for job_id in throughputs:
            allocation[job_id] = {}
            for worker_id in throughputs[job_id]:
                if job_id in self._allocation and self._allocation[job_id] == worker_id:
                    allocation[job_id][worker_id] = 1.0
                else:
                    allocation[job_id][worker_id] = 0.0
        return allocation
