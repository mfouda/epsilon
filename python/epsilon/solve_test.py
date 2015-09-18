
import logging

import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon.problems import problems_test

# TODO(mwytock): Fix these problems to achieve higher accuracy
REL_TOL = {
    "basis_pursuit": 1e-1
}

def solve_problem(problem_instance):
    problem = problem_instance.create()

    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    logging.debug(problem_instance.name)
    solve.solve(problem)
    obj1 = problem.objective.value

    rtol = REL_TOL.get(problem_instance.name, 1e-2)
    np.testing.assert_allclose(obj0, obj1, rtol=rtol, atol=1e-4)

def test_solve():
    for problem in problems_test.get_problems():
        yield solve_problem, problem
