import sys; sys.path.append("..")
from throughput_estimator import ThroughputEstimator
from job_table import JobTable
import utils

import numpy as np
import unittest

class TestThroughputEstimator(unittest.TestCase):

    def setUp(self):
        self._oracle_throughputs = \
            utils.read_all_throughputs_json_v2('../oracle_throughputs_v2.json')
        self._worker_types = ['k80', 'p100', 'v100']
        self._job_types = [(JobTable[i].model, 1) for i in range(len(JobTable))]

    def test_no_estimation(self):
        num_reference_models = len(self._job_types)
        profiling_percentage = 1.0
        estimator = ThroughputEstimator(self._oracle_throughputs,
                                        self._worker_types, self._job_types,
                                        num_reference_models,
                                        profiling_percentage)

        for job_type in self._job_types:
            estimated_throughputs = estimator.estimate_throughputs(job_type)
            for worker_type in estimated_throughputs:
                oracle_throughputs = \
                    self._oracle_throughputs[worker_type][job_type]
                for other_job_type in estimated_throughputs[worker_type]:
                    estimated_throughput = \
                        estimated_throughputs[worker_type][other_job_type]
                    oracle_throughput = oracle_throughputs[other_job_type][0]
                    assert(np.isclose(estimated_throughput, oracle_throughput))

    def test_estimation(self):
        num_reference_models = 16
        profiling_percentage = 0.6
        estimator = ThroughputEstimator(self._oracle_throughputs,
                                        self._worker_types, self._job_types,
                                        num_reference_models,
                                        profiling_percentage)
        m = len(self._worker_types)
        n = len(self._job_types)
        flattened_estimated_throughputs = np.zeros((m * n))
        flattened_oracle_throughputs = np.zeros((m * n))
        for job_type in self._job_types:
            estimated_throughputs = estimator.estimate_throughputs(job_type)
            for i, worker_type in enumerate(self._worker_types):
                oracle_throughputs = \
                    self._oracle_throughputs[worker_type][job_type]
                isolated_throughput = oracle_throughputs['null']
                for j, other_job_type in enumerate(self._job_types):
                    offset = i * n + j
                    estimated_throughput = \
                        estimated_throughputs[worker_type][other_job_type]
                    oracle_throughput = oracle_throughputs[other_job_type][0]
                    flattened_estimated_throughputs[offset] = \
                        estimated_throughput / isolated_throughput
                    flattened_oracle_throughputs[offset] = \
                        oracle_throughput / isolated_throughput
            delta = \
                flattened_oracle_throughputs - flattened_estimated_throughputs
            print('%s: %f' % (job_type, np.linalg.norm(delta)))

if __name__=='__main__':
    unittest.main()
