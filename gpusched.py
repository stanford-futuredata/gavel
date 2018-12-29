import numpy as np


class Problem:
    def __init__(self, a):
        """
        a: 2D matrix where a[i][j] is the progress rate of user i on resource j
        """
        assert len(a.shape) == 2
        self.a = a

    @property
    def n(self):
        """Number of users in the problem"""
        return self.a.shape[0]

    @property
    def m(self):
        """Number of resources in the problem"""
        return self.a.shape[1]

    def __repr__(self):
        return "Problem(a=%s)" % self.a


class Solution:
    def __init__(self, problem, x):
        """
        x: a 2D matrix where x[i][j] is the fraction of resource j allocated to user i
        """
        assert len(x.shape) == 2
        self.problem = problem
        self.x = x

    def __repr__(self):
        return "Solution(x=%s)" % self.x

    @property
    def user_rates(self):
        return (self.x * self.problem.a).sum(axis=1)


def solve_isolated(problem):
    """
    This algorithm splits all resources equally between all users, giving the isolation baseline.
    """
    x = np.full(problem.a.shape, 1.0 / problem.n)
    return Solution(problem, x)


def solve_max_throughput(problem):
    """
    This algorithm tries to maximize total throughput by assigning each resource to the user who
    will receive the highest rate from it. If multiple users are tied in their rate from a
    resource, we split it equally between them.
    """
    col_maxes = problem.a.max(axis=0)
    is_max = problem.a == col_maxes
    num_equal_to_max = is_max.sum(axis=0)
    scale = 1.0 / num_equal_to_max
    x = is_max * scale
    return Solution(problem, x)


def print_solution(name, solution):
    print("%s assignments:\n%s" % (name, solution.x))
    rates = solution.user_rates
    total_rate = rates.sum()
    print("%s rates:\n %s (total: %f)" % (name, solution.user_rates, total_rate))


def explore_problem(a):
    a = np.array(a, dtype=np.float64)
    print("Exploring problem:\n%s" % a)
    problem = Problem(a)
    print_solution("Isolated", solve_isolated(problem))
    print_solution("Max throughput", solve_max_throughput(problem))
    print()


def main():
    explore_problem([[1., 2.], [2., 1.]])
    explore_problem([[1., 2.], [1., 1.]])
    explore_problem([[1., 2.], [2., 1.], [2., 2.]])


if __name__ == "__main__":
    main()
