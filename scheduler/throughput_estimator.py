import copy
import json
import matrix_completion
import numpy as np
import random
import sys
import warnings

DEFAULT_MATRIX_COMPLETION_K = 10
DEFAULT_MATRIX_COMPLETION_MU = 1e-2

def cosine_distance(a, b):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ThroughputEstimator:
    def __init__(self, oracle_throughputs, worker_types, job_types,
                 num_reference_job_types, profiling_percentage, seed=0):
        self._rng = random.Random()
        self._rng.seed(seed)
        self._oracle_throughputs = oracle_throughputs
        self._worker_types = worker_types
        self._job_types = job_types
        self._m = len(worker_types)
        self._n = len(job_types)
        self._profiling_percentage = profiling_percentage
        self._get_normalized_throughputs()
        self._get_reference_throughputs(num_reference_job_types)

    def _get_normalized_throughputs(self):
        m = self._m
        n = self._n
        self._normalized_throughputs = np.zeros((n, m * n), dtype=np.float32)
        for i, job_type in enumerate(self._job_types):
            for j, worker_type in enumerate(self._worker_types):
                per_worker_oracle_throughputs = \
                    self._oracle_throughputs[worker_type][job_type]
                for k, other_job_type in enumerate(self._job_types):
                    self._normalized_throughputs[i, j*n+k] = \
                        per_worker_oracle_throughputs[other_job_type][0] /\
                            per_worker_oracle_throughputs['null']
        assert(np.min(self._normalized_throughputs) >= 0 and
               np.max(self._normalized_throughputs) <= 1.0)

    def _get_reference_throughputs(self, num_reference_job_types):
        m = self._m
        n = self._n
        reference_job_types_idx = \
            sorted(self._rng.sample(list(range(len(self._job_types))),
                                    num_reference_job_types))
        self._reference_job_types = \
            [self._job_types[i] for i in reference_job_types_idx]
        self._reference_throughputs = \
            self._normalized_throughputs[reference_job_types_idx]
        reference_job_types_column_idx = []
        for i in range(m):
            reference_job_types_column_idx += \
                map(lambda x: x + i * n,
                    reference_job_types_idx)
        self._reference_throughputs = \
            self._reference_throughputs[:,reference_job_types_column_idx]

        if len(self._reference_job_types) == len(self._job_types):
            assert(self._reference_job_types == self._job_types)
            assert((self._normalized_throughputs == \
                    self._reference_throughputs).all())

    def _profile_jobs(self, true_job_type):
        true_job_type_idx = self._job_types.index(true_job_type)
        profiled_jobs = {}
        for i, worker_type in enumerate(self._worker_types):
            profiled_jobs[worker_type] = {}
            for j, reference_job_type in enumerate(self._reference_job_types):
                r = self._rng.uniform(0, 1)
                if r <= self._profiling_percentage:
                    offset = i * len(self._reference_job_types) + j
                    profiled_jobs[worker_type][reference_job_type] = \
                        self._normalized_throughputs[true_job_type_idx][offset]
        return profiled_jobs

    def match_job_to_reference_job(self, true_job_type):
        """Uses matrix completion to match a job to a reference job type.

        Uses a subset of measured data points to match an unseen job to
        a reference job type measured offline.
        """
        profiled_jobs = self._profile_jobs(true_job_type)

        # Initialize the throughputs matrix using the pre-measured
        # reference throughputs.
        throughputs_matrix = np.zeros((self._reference_throughputs.shape[0] + 1,
                                       self._reference_throughputs.shape[1]),
                                       dtype=np.float32)
        throughputs_matrix[:-1,:] = self._reference_throughputs

        # Initialize the mask, setting the mask of all pre-measured values to 1.
        mask = np.zeros(throughputs_matrix.shape, dtype=np.float32)
        mask[:-1,:] = 1

        # Fill in measured data points.
        for i, worker_type in enumerate(sorted(profiled_jobs.keys())):
            for j, reference_job_type in enumerate(self._reference_job_types):
                if reference_job_type in profiled_jobs[worker_type]:
                    offset = i * len(self._reference_job_types) + j
                    throughputs_matrix[-1][offset] = \
                        profiled_jobs[worker_type][reference_job_type]
                    mask[-1][offset] = 1

        if np.min(mask) == 0:
            # Run matrix completion algorithm if there are values to estimate.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                k = DEFAULT_MATRIX_COMPLETION_K
                mu = DEFAULT_MATRIX_COMPLETION_MU
                try:
                    estimated_throughputs = \
                        matrix_completion.pmf_solve(throughputs_matrix,
                                                    mask, k=k, mu=mu),
                    throughputs_matrix = \
                        np.where(mask, throughputs_matrix,
                                 np.clip(estimated_throughputs, 0, 1))[0]
                except np.linalg.LinAlgError as e:
                    print('WARNING: could not estimate throughputs!',
                          file=sys.stderr)
                    print(e, file=sys.stderr)
                    return self._rng.choice(self._reference_job_types)
        else:
            print('WARNING: Did not run matrix completion as mask is complete',
                  file=sys.stderr)

        # Measure the distance from the new row to every other row and find
        # the row with the smallest distance.
        distances = []
        if np.linalg.norm(throughputs_matrix[-1]) == 0:
            print('WARNING: Norm of predicted throughputs is 0!')
            return self._rng.choice(self._reference_job_types)
        for i, reference_job_type in enumerate(self._reference_job_types):
            distance = cosine_distance(throughputs_matrix[i],
                                       throughputs_matrix[-1])
            distances.append((reference_job_type, distance))
        distances.sort(key=lambda x: x[1])
        predicted_job_type = distances[0][0]

        return predicted_job_type

    def get_reference_throughputs(self):
        m = self._m
        n = len(self._reference_job_types)
        reference_throughputs = {}
        for i, worker_type in enumerate(self._worker_types):
            reference_throughputs[worker_type] = {}
            for j, reference_job_type in enumerate(self._reference_job_types):
                reference_throughputs[worker_type][reference_job_type] = {}
                for k, other_reference_job_type in enumerate(self._reference_job_types):
                    offset = i * n + k
                    inverse_offset = i * n + j
                    reference_throughputs[worker_type][reference_job_type][other_reference_job_type] = \
                        [self._reference_throughputs[j, offset],
                         self._reference_throughputs[k, inverse_offset]]
        return reference_throughputs
