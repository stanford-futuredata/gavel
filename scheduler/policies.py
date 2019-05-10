import cvxpy as cp
import numpy as np


class Policy:

    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    def flatten(self, d, cluster_spec):
        """Converts a 2-level dict to a NumPy array."""

        job_id_pairs = sorted(list(d.keys()))
        if len(job_id_pairs) == 0:
            return None, None
        worker_types = sorted(list(d[job_id_pairs[0]].keys()))
        self._num_workers = [cluster_spec[worker_type] for worker_type in worker_types]
        if len(worker_types) == 0:
            return None, None
        m = []
        for job_id in job_id_pairs:
            m_row = []
            for worker_type in worker_types:
                m_row.append(d[job_id][worker_type])
            m.append(m_row)
        return np.array(m), (job_id_pairs, worker_types)

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

    def __init__(self):
        self._name = 'Isolated'

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        return super().unflatten(np.full((m, n), 1.0 / (m * n)), index)


class MaximumThroughputPolicy(Policy):

    def __init__(self):
        self._name = 'MaximumThroughput'

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(cp.sum(cp.sum(cp.multiply(throughputs, x),
                                              axis=1)))
        constraints = [
            x >= 0,
            cp.sum(x, axis=0) <= 4,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        assert cvxprob.status == "optimal"
        return super().unflatten(x.value.clip(min=0.0), index)


class KSPolicy(Policy):

    def __init__(self):
        self._name = 'KS'

    def get_allocation(self, unflattened_throughputs, cluster_spec={'k80': 4, 'p100': 4, 'v100': 4}):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        scale = 1.0 / throughputs.sum(axis=1)
        throughputs = throughputs * scale.reshape(m, 1)

        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(cp.min(cp.sum(cp.multiply(throughputs, x),
                                              axis=1)))
        constraints = [
            x >= 0,
            cp.sum(x, axis=0) <= 4,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        assert cvxprob.status == "optimal"

        return super().unflatten(x.value.clip(min=0.0), index)


class KSPolicyWithPacking(Policy):

    def __init__(self):
        self._name = 'KS_Packing'

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
        single_job_ids = []
        for job_id in job_ids:
            if not job_id.is_pair():
                single_job_ids.append(job_id)

        # Compute normalizing factor for each individual job, this normalizing
        # factor will be used to normalize throughputs for the same job in job
        # combinations as well.
        normalizing_factors = {}
        for single_job_id in single_job_ids:
            normalizing_factor = 0.0
            for worker_type in worker_types:
                normalizing_factor += d[single_job_id][worker_type]
            normalizing_factors[single_job_id] = normalizing_factor

        if len(worker_types) == 0:
            return None, None, None

        all_m = []
        masks = []
        # Compute the throughput matrix and mask for each individual job.
        for single_job_id in single_job_ids:
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
                    if job_id in single_job_ids:
                        if job_id != single_job_id:
                            m_row.append(0.0)
                            mask_row.append(0.0)
                        else:
                            m_row.append(d[job_id][worker_type])
                            mask_row.append(1.0)
                    else:
                        if not single_job_id.overlaps_with(job_id):
                            m_row.append(0.0)
                            mask_row.append(1.0)
                        else:
                            # Find the index of the job of interest in the job
                            # combination tuple.
                            index = job_id.as_tuple().index(single_job_id[0])
                            throughputs = d[job_id][worker_type]
                            m_row.append(d[job_id][worker_type][index])
                            mask_row.append(1.0)
                m.append(m_row)
                mask.append(mask_row)
            # Normalize
            all_m.append(np.array(m) / normalizing_factors[single_job_id])
            masks.append(np.array(mask))
        return all_m, masks, (job_ids, single_job_ids, worker_types)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_id_combinations, single_job_ids, worker_types) = index
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
        self._name = 'FIFO'
        self._allocation = {}
        self._queue = []

    def get_allocation(self, throughputs, cluster_spec={'k80': 4, 'p100': 4, 'v100': 4}):
        # New Job ID; put on queue to schedule.
        job_id = None
        for job_id in throughputs:
            if job_id not in self._allocation and job_id not in self._queue:
                self._queue.append(job_id)

        # Old Job ID that has been removed; schedule job from queue.
        job_ids = list(self._allocation.keys())
        for job_id in job_ids:
            if job_id not in throughputs:
                worker_type = self._allocation[job_id]
                del self._allocation[job_id]
                if len(self._queue) > 0:
                    job_id_to_schedule = self._queue.pop(0)
                    self._allocation[job_id_to_schedule] = worker_type

        # worker_types_seen keeps track of all workers that have been assigned
        # jobs.
        worker_types_seen = {}
        for job_id in self._allocation:
            worker_type = self._allocation[job_id]
            if worker_type not in worker_types_seen:
                worker_types_seen[worker_type] = 0
            worker_types_seen[worker_type] += 1

        # Try to allocation all queued job IDs on available workers.
        job_ids = list(throughputs.keys())
        if len(job_ids) > 0:
            job_id = job_ids[0]
            for worker_type in throughputs[job_id]:
                if (worker_type not in worker_types_seen or
                    worker_types_seen[worker_type] < cluster_spec[worker_type]):
                    if len(self._queue) > 0:
                        job_id_to_schedule = self._queue.pop(0)
                        self._allocation[job_id_to_schedule] = worker_type

        # Construct output allocation.
        allocation = {}
        for job_id in throughputs:
            allocation[job_id] = {}
            for worker_type in throughputs[job_id]:
                if (job_id in self._allocation and
                    self._allocation[job_id] == worker_type):
                    allocation[job_id][worker_type] = 1.0
                else:
                    allocation[job_id][worker_type] = 0.0
        return allocation
