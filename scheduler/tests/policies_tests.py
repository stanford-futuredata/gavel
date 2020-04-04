import sys; sys.path.append("..")
from job_id_pair import JobIdPair
from policies import finish_time_fairness, max_min_fairness, max_sum_throughput
import unittest

class TestPolicies(unittest.TestCase):

    def test_max_min_fairness(self):
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        unflattened_priority_weights = {0: 1, 1: 1}
        times_since_start = {0: 0, 1: 0}
        num_steps_remaining = {0: 300, 1: 500}
        cluster_spec = {
            'v100': 1,
            'p100': 2,
            'k80': 3
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              cluster_spec)

    def test_finish_time_fairness(self):
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        unflattened_priority_weights = {0: 1, 1: 1}
        times_since_start = {0: 0, 1: 0}
        num_steps_remaining = {0: 300, 1: 500}
        cluster_spec = {
            'v100': 1,
            'p100': 2,
            'k80': 3
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)

    def test_finish_time_fairness_with_packing(self):
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPacking(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        unflattened_priority_weights = {JobIdPair(0, None): 1,
                                        JobIdPair(1, None): 1}
        times_since_start = {JobIdPair(0, None): 0, JobIdPair(1, None): 0}
        num_steps_remaining = {JobIdPair(0, None): 300, JobIdPair(1, None): 500}
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)

    def test_throughput_sum(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs=None)

    def test_throughput_sum_normalized_by_cost(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 3.1, 'p100': 2.0, 'k80': 0.8})

    def test_throughput_sum_normalized_by_cost_with_SLOs(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerfSLOs(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 3.1, 'p100': 2.0, 'k80': 0.8},
                              SLOs={0: 1000}, num_steps_remaining={0: 100})

    def test_throughput_sum_normalized_by_cost_with_packing_and_SLOs(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 3.1, 'p100': 2.0, 'k80': 0.8},
                              SLOs={JobIdPair(0, None): 1000},
                              num_steps_remaining={JobIdPair(0, None): 100})

    def test_throughput_sum_normalized_by_cost_with_packing_and_SLOs_v2(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 1.0, 'p100': 1.0, 'k80': 1.0},
                              SLOs={JobIdPair(0, None): 1000},
                              num_steps_remaining={JobIdPair(0, None): 100})


if __name__=='__main__':
    unittest.main()
