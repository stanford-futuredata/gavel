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
            predicted_job_type = estimator.match_job_to_reference_job(job_type)
            assert(job_type == predicted_job_type)

        reference_throughputs = estimator.get_reference_throughputs()
        for worker_type in self._worker_types:
            per_worker_reference_throughputs = \
                reference_throughputs[worker_type]
            per_worker_oracle_throughputs = \
                self._oracle_throughputs[worker_type]
            for job_type in self._job_types:
                for other_job_type in self._job_types:
                    oracle_throughput = \
                        per_worker_oracle_throughputs[job_type][other_job_type],
                    isolated_throughput = \
                        [per_worker_oracle_throughputs[job_type]['null'],
                        per_worker_oracle_throughputs[other_job_type]['null']]
                    estimated_throughput = \
                        np.multiply(isolated_throughput,
                                    per_worker_reference_throughputs[job_type][other_job_type])
                    assert(np.all(np.isclose(oracle_throughput,
                                             estimated_throughput)))

    def test_estimation(self):
        num_reference_models = 16
        profiling_percentage = 0.6
        estimator = ThroughputEstimator(self._oracle_throughputs,
                                        self._worker_types, self._job_types,
                                        num_reference_models,
                                        profiling_percentage)
        reference_models = set()
        for job_type in self._job_types:
            predicted_job_type = estimator.match_job_to_reference_job(job_type)
            reference_models.add(predicted_job_type)
        assert(len(reference_models) <= num_reference_models)

if __name__=='__main__':
    unittest.main()
