
import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon.problems import problems_test

def solve_problem(problem_instance):
    problem = problem_instance.create()

    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    solve.solve(problem)
    obj1 = problem.objective.value

    np.testing.assert_allclose(obj0, obj1, rtol=1e-2, atol=1e-4)

EXCLUDE = set(["tv_smooth"])
def test_solve():
    for problem in problems_test.get_problems():
        if problem.name not in EXCLUDE:
            yield solve_problem, problem
