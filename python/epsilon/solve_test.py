
import cvxpy as cp
import numpy as np

import epsilon
from epsilon import status_pb2
from epsilon.problems import problems_test

def solve_problem(problem):
    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    epsilon.solve(problem)
    obj1 = problem.objective.value

    np.testing.assert_allclose(obj0, obj1, rtol=1e-2, atol=1e-4)

def test_solve():
    for problem in problems_test.get_test_problems():
        yield solve_problem, problem
