import cvxpy as cp
import numpy as np


class Problem:
    def __init__(self, a, throughput_masks=None):
        """
        a: 2D matrix where a[i][j] is the progress rate of user i on resource j
        """
        assert len(a.shape) == 2
        self.a = a
        self.throughput_masks = throughput_masks

    @property
    def n(self):
        """Number of users in the problem."""
        return self.a.shape[0]

    @property
    def m(self):
        """Number of resources in the problem."""
        return self.a.shape[1]

    def user_rates(self, solution):
        """Return the progress rate achieved by each user in a solution."""
        if self.throughput_masks is None:
            return (self.a * solution.x).sum(axis=1)
        return np.array([np.multiply(np.multiply(self.a, throughput_mask), solution.x).sum()
                         for throughput_mask in self.throughput_masks])

    def __repr__(self):
        return "Problem(a=%s)" % self.a


class Solution:
    def __init__(self, x):
        """
        x: 2D matrix where x[i][j] is the fraction of resource j allocated to user i
        """
        assert len(x.shape) == 2
        self.x = x

    def __repr__(self):
        return "Solution(x=%s)" % self.x


def solve_isolated(problem):
    """
    This algorithm splits all resources equally between all users, giving the isolation baseline.
    """
    x = np.full(problem.a.shape, 1.0 / problem.n)
    return Solution(x)


def solve_ks_with_packing(problem, throughput_masks):
    """
    This algorithm tries to equalize the users' normalized throughputs (i.e. the
    Kalai-Smorodinsky bargaining solution, similar to DRF).
    """
    a = problem.a
    shape = a.shape
    a = a.flatten(order='F')
    x = cp.Variable(a.shape)
    all_a_masked = []
    for throughput_mask in throughput_masks:
        throughput_mask = np.array(throughput_mask)
        a_masked = np.multiply(throughput_mask.flatten(order='F'), a)
        all_a_masked.append(a_masked)
    all_a_masked = np.array(all_a_masked)
    objective = cp.Maximize(cp.min(cp.matmul(all_a_masked, x)))
    constraints = [
        x >= 0,
        cp.sum(cp.reshape(x, shape), axis=0) == 1,
    ]
    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve()
    assert cvxprob.status == "optimal"
    x_sol = np.reshape(x.value, shape, order='F')
    return Solution(x_sol)


def explore_problem(a, throughput_masks, num_users):
    """
    Explore a scheduling problem, printing the various solutions.
    """
    a = np.array(a, dtype=np.float64)
    problem = Problem(a, throughput_masks)

    print("For problem %s:" % str(a).replace('\n', ''))

    problem_only_users = Problem(a[:num_users])
    print_solution("Isolated (without packing)", problem_only_users,
                   solve_isolated(problem_only_users))
    print_solution("Isolated (with packing)", problem, solve_isolated(problem))
    print_solution("Kalai-Smorodinsky (with packing)", problem, solve_ks_with_packing(
        problem, throughput_masks=throughput_masks))
    print()


def print_solution(name, problem, solution):
    """
    Pretty-print and analyze a problem solution.
    """
    print("%s solution:" % name)
    print("  assignments: %s" % str(solution.x).replace('\n', ''))
    rates = problem.user_rates(solution)
    print("  user rates: %s (total: %.3g)" % (rates, rates.sum()))


def main():
    np.set_printoptions(precision=4, suppress=True)
    explore_problem(
        [[100.0, 100.0], [100.0, 100.0], [170.0, 170.0]],
        throughput_masks=[np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]]),
                          np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])],
        num_users=2)
    explore_problem(
        [[100.0, 100.0], [100.0, 100.0], [95.0, 95.0]],
        throughput_masks=[np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]]),
                          np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])],
        num_users=2)


if __name__ == "__main__":
    main()
