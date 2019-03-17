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
    def get_allocation_flattened(self, throughputs):
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
        return x.value.clip(min=0.0)

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        return super().unflatten(self.get_allocation_flattened(throughputs),
                                 index)


class KSPolicyNormalized(KSPolicy):
    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        scale = 1.0 / throughputs.sum(axis=1)
        throughputs = throughputs * scale.reshape(m, 1)
        return super().unflatten(super().get_allocation_flattened(throughputs),
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

        worker_ids_seen = set()
        for job_id in self._allocation:
            worker_id = self._allocation[job_id]
            worker_ids_seen.add(worker_id)

        job_ids = list(throughputs.keys())
        if len(job_ids) > 0:
            job_id = job_ids[0]
            for worker_id in throughputs[job_id]:
                if worker_id not in worker_ids_seen:
                    if len(self._queue) > 0:
                        job_id_to_schedule = self._queue.pop(0)
                        self._allocation[job_id_to_schedule] = worker_id

        allocation = {}
        for job_id in throughputs:
            allocation[job_id] = {}
            for worker_id in throughputs[job_id]:
                if job_id in self._allocation and self._allocation[job_id] == worker_id:
                    allocation[job_id][worker_id] = 1.0
                else:
                    allocation[job_id][worker_id] = 0.0
        return allocation
