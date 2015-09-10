
import cvxpy as cp
import numpy as np

import epsilon
from epsilon.problems import covsel
from epsilon.problems import lasso
from epsilon.problems import tv_smooth
from epsilon.problems.problem_instance import ProblemInstance

PROBLEMS = [
    ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    ProblemInstance("lasso", lasso.create, dict(m=5, n=10)),
    ProblemInstance("tv_smooth", tv_smooth.create, dict(n=10, lam=1)),
]

def solve_problem(problem):
    cvxpy_prob = problem.create()

    cvxpy_prob.solve(solver=cp.SCS)
    obj0 = cvxpy_prob.objective.value

    epsilon.solve(cvxpy_prob)
    obj1 = cvxpy_prob.objective.value

    np.testing.assert_allclose(obj0, obj1, rtol=1e-2, atol=1e-4)


def test_problems():
    for problem in PROBLEMS:
        yield solve_problem, problem
