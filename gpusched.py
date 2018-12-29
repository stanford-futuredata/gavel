import cvxpy as cp
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

    @property
    def normalized(self):
        """
        Return a normalized version of the problem where each user's progress rate would
        be 1.0 if it could use all the resources.
        """
        scale = 1.0 / self.a.sum(axis=1)
        new_a = self.a * scale.reshape(self.n, 1)
        return Problem(new_a)

    def __repr__(self):
        return "Problem(a=%s)" % self.a


class Solution:
    def __init__(self, problem, x):
        """
        problem: the problem we are solving
        x: 2D matrix where x[i][j] is the fraction of resource j allocated to user i
        """
        assert len(x.shape) == 2
        self.problem = problem
        self.x = x

    def __repr__(self):
        return "Solution(x=%s)" % self.x

    @property
    def user_rates(self):
        return (self.x * self.problem.a).sum(axis=1)

    @property
    def normalized_user_rates(self):
        return (self.x * self.problem.normalized.a).sum(axis=1)


def solve_isolated(problem):
    """
    This algorithm splits all resources equally between all users, giving the isolation baseline.
    """
    x = np.full(problem.a.shape, 1.0 / problem.n)
    return Solution(problem, x)


def solve_max_throughput(problem, normalize=True):
    """
    This algorithm tries to maximize total throughput by assigning each resource to the user who
    will receive the highest rate from it. If multiple users are tied in their rate from a
    resource, we split it equally between them.
    """
    a = problem.a
    if normalize:
        a = problem.normalized.a

    col_maxes = a.max(axis=0)
    is_max = a == col_maxes
    num_equal_to_max = is_max.sum(axis=0)
    scale = 1.0 / num_equal_to_max
    x = is_max * scale
    return Solution(problem, x)


def solve_isolated_max_throughput(problem):
    """
    This algorithm tries to maximize total normalized throughput subject to giving each user
    at least as much as they'd get in the isolated solution.
    """
    a = problem.normalized.a
    x = cp.Variable(a.shape)
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)))
    constraints = [
        x >= 0,
        cp.sum(x, axis=0) <= 1,
        cp.sum(cp.multiply(a, x), axis=1) >= 1.0 / problem.n
    ]
    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve()
    return Solution(problem, x.value)


def explore_problem(a):
    """
    Explore a scheduling problem, printing the various solutions.
    """
    a = np.array(a, dtype=np.float64)
    problem = Problem(a)
    print("For problem %s:" % str(a).replace('\n', ''))
    print("  (normalized: %s)" % str(problem.normalized.a).replace('\n', ''))
    print_solution("Isolated", solve_isolated(problem))
    #print_solution("Unnormalized max throughput", solve_max_throughput(problem, False))
    print_solution("Max throughput", solve_max_throughput(problem))
    print_solution("Isolated max throughput", solve_isolated_max_throughput(problem))
    print()


def print_solution(name, solution):
    """
    Pretty-print and analyze a problem solution.
    """
    print("%s solution:" % name)
    print("  assignments: %s" % str(solution.x).replace('\n', ''))
    #rates = solution.user_rates
    #print("  user rates: %s (total: %.3g)" % (rates, rates.sum()))
    norm_rates = solution.normalized_user_rates
    print("  normalized rates: %s (total: %.3g)" % (norm_rates, norm_rates.sum()))


def main():
    np.set_printoptions(precision=3, suppress=True)
    explore_problem([[1., 2.], [2., 1.]])
    explore_problem([[1., 2.], [1., 1.]])
    explore_problem([[1., 2.], [10., 10.]])
    explore_problem([[1., 2.], [2., 1.], [1., 1.]])
    print("Example showing lack of strategy-proofness (P3 gets more by faking their demand):\n")
    explore_problem([[2., 1.], [1., 1.], [1., 1.]])
    explore_problem([[2., 1.], [1., 1.], [1., 3.]])


if __name__ == "__main__":
    main()
