
import logging

import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.problems import basis_pursuit
from epsilon.problems import covsel
from epsilon.problems import group_lasso
from epsilon.problems import huber
from epsilon.problems import lasso
from epsilon.problems import logreg_l1
from epsilon.problems import lp
from epsilon.problems import least_abs_dev
from epsilon.problems import quantile
from epsilon.problems import tv_1d
from epsilon.problems import tv_smooth
from epsilon.problems.problem_instance import ProblemInstance

# These problems need a higher relative accuracy for some reason
REL_TOL = {
    "basis_pursuit": 1e-3,
    "ls_mae": 1e-3,
    "group_lasso": 1e-3
}

# TODO(mwytock): Need L1/Linf proximal operator
# ProblemInstance("quantile", quantile.create, dict(m=20, n=10, k=3))
#
# TODO(mwytock): Logistic prox (or cone reduction) not implemented. Also need to
# support more than one equality constraint
# ProblemInstance("logreg_l1", logreg_l1.create, dict(m=5, n=10))
#
# TODO(mwytock): gives wrong answer, likely has to do with linearized ADMM?
# ProblemInstance("tv_1d", tv_1d.create, dict(n=10)),

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    ProblemInstance("group_lasso", group_lasso.create, dict(m=15, ni=5, K=10)),
    ProblemInstance("huber", huber.create, dict(m=20, n=10)),
    ProblemInstance("lasso", lasso.create, dict(m=5, n=10)),
    ProblemInstance("lp", lp.create, dict(m=10, n=20)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=10, n=5)),
    ProblemInstance("tv_smooth", tv_smooth.create, dict(n=10, lam=1)),
]

def solve_problem(problem_instance):
    problem = problem_instance.create()

    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    logging.debug(problem_instance.name)
    params = solver_params_pb2.SolverParams()
    params.rel_tol = REL_TOL.get(problem_instance.name, 1e-2)
    solve.solve(problem, params)
    obj1 = problem.objective.value

    np.testing.assert_allclose(obj0, obj1, rtol=1e-2, atol=1e-4)

def test_solve():
    for problem in PROBLEMS:
        yield solve_problem, problem
