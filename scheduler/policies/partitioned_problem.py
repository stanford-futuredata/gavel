import copy
import random


class PartitionedProblem:
    def __init__(self, policy, k):
        self._policy_instances = [copy.deepcopy(policy) for _ in range(k)]
        self._k = k
        self._name = policy._name

    def get_allocation(self, *args, **kwargs):
        args_list = list(args)
        throughputs = args_list[0]
        cluster_spec = args_list[-1]

        sub_problem_cluster_spec = {x: cluster_spec[x] // self._k
                                    for x in cluster_spec}

        job_to_sub_problem_assignment = {}
        job_ids = []
        for job_id in throughputs:
            if not job_id.is_pair():
                job_ids.append(job_id)
        for i, job_id in enumerate(job_ids):
            job_to_sub_problem_assignment[job_id[0]] = \
                random.randint(0, self._k-1)

        sub_problem_throughputs = []
        for i in range(self._k):
            sub_problem_throughputs.append({})
            for job_id in throughputs:
                if (job_to_sub_problem_assignment[job_id[0]] == i) and (
                    not job_id.is_pair() or (job_to_sub_problem_assignment[job_id[1]] == i)):
                    sub_problem_throughputs[-1][job_id] = copy.copy(
                        throughputs[job_id])

        full_allocation = {}
        for i in range(self._k):
            args_list_sub_problem = copy.deepcopy(args_list[1:])
            args_list_sub_problem[-1] = sub_problem_cluster_spec
            args_list_sub_problem = [sub_problem_throughputs[i]] + args_list_sub_problem
            print(args_list_sub_problem)
            sub_problem_allocation = self._policy_instances[i].get_allocation(
                *args_list_sub_problem, **kwargs)
            for job_id in sub_problem_allocation:
                full_allocation[job_id] = sub_problem_allocation[job_id]

        return full_allocation
