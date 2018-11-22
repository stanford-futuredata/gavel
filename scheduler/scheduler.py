import numpy as np
import cvxpy

n = 6  # Number of applications.
m = 3  # Number of machines.

# (numbers adapted from
# https://docs.google.com/spreadsheets/d/17j0WkjEaZFz_Wv5e4ZuPjaYhb2TRD85BYnnM9ATmU9Q/edit?usp=sharing).
throughputs = np.array([
    [1, 1, 1, 1, 1, 1],
    [5, 4.5, 3.8, 1.6, 3.8, 1.2],
    [7.5, 8, 6.8, 2, 4.9, 2.0]
])
A = 1.0 / throughputs  # Time estimates for application i on machine j
print("Time estimates:", A)

# Variable definitions.
X = cvxpy.Variable((m, n))  # Variable encoding whether application i runs on machine j.
y_max = cvxpy.Variable()

# Constraints (X_{ij} should be an integer between 0 and 1, but ignore the integer
# part for now).
X_lower_bound_constraint = X >= 0.
X_upper_bound_constraint = X <= 1.
sum_constraint = cvxpy.sum(X, axis=0) == 1 # Only one machine active for an application.
y_j = cvxpy.sum(cvxpy.multiply(X, A), axis=1) # Total time spent on applications on each machine.
y_max_constraint = y_max >= y_j

constraints = [X_lower_bound_constraint, X_upper_bound_constraint,
               sum_constraint, y_max_constraint]

problem = cvxpy.Problem(cvxpy.Minimize(y_max), constraints=constraints)
problem.solve()
floating_point_placements = X.value
int_placements = np.round(floating_point_placements)
print("Application placements:", np.array(np.round(floating_point_placements),
                                          dtype=np.bool))
print("Total time:", max(np.sum(np.multiply(int_placements, A), axis=1)))
