import sys; sys.path.append("..")
from policies import hierarchical


def test_two_level_single_pass_hierarchical():
    policy = hierarchical.TwoLevelSinglePassHierarchicalPolicy(
        solver='ECOS')
    unflattened_throughputs = {
        0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
        1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
        2: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
    }
    scale_factors = {
        0: 1,
        1: 1,
        2: 1
    }
    unflattened_priority_weights = {'A': 1, 'B': 3}
    job_to_entity_mapping = {0: 'A', 1: 'B', 2: 'B'}
    cluster_spec = {
        'v100': 1,
        'p100': 2,
        'k80': 3
    }
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       job_to_entity_mapping,
                                       cluster_spec)
    return allocation


if __name__ == '__main__':
    allocation = test_two_level_single_pass_hierarchical()
    print(allocation)