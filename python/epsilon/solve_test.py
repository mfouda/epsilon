

import logging

import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.problems import *
from epsilon.problems.problem_instance import ProblemInstance as Prob

# Override accuracy settings
REL_TOL = {}

# Add a multiclass classification problem w/ hinge loss
#
# Need 2D convolution operators or better splitting?
# ProblemInstance("tv_denoise", tv_denoise.create, dict(n=10, lam=1)),
#
# Huge expression tree. consider way to do graph problems?
# ProblemInstance("map_inference", map_inference.create, dict(n=10)),
#
# Need to fix ScaledZoneProx family functions to make canonicalize robust
# ProblemInstance("robust_svm", robust_svm.create, dict(m=20, n=10, k=3)),
#
# Need to fix case where ATA is not identity
# ProblemInstance("portfolio", portfolio.create, dict(m=5, n=10)),

PROBLEMS = [
    Prob("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    Prob("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    Prob("group_lasso", group_lasso.create, dict(m=15, ni=5, K=10)),
    Prob("hinge_l1", hinge_l1.create, dict(m=5, n=20, rho=0.1)),
    Prob("hinge_l1_sparse", hinge_l1.create, dict(m=5, n=20, mu=0.1)),
    Prob("hinge_l2", hinge_l2.create, dict(m=20, n=10, rho=1)),
    Prob("hinge_l2_sparse", hinge_l2.create, dict(m=20, n=20, rho=1, mu=0.1)),
    Prob("huber", huber.create, dict(m=20, n=10)),
    Prob("lasso", lasso.create, dict(m=5, n=20, rho=0.1)),
    Prob("lasso_sparse", lasso.create, dict(m=5, n=20, rho=0.1, mu=0.1)),
    Prob("least_abs_dev", least_abs_dev.create, dict(m=10, n=5)),
    Prob("logreg_l1", logreg_l1.create, dict(m=5, n=10)),
    Prob("logreg_l1_sparse", logreg_l1.create, dict(m=5, n=20, mu=0.1)),
    Prob("lp", lp.create, dict(m=10, n=20)),
    Prob("mnist", mnist.create, dict(data=mnist.DATA_TINY, n=10)),
    Prob("mv_lasso", lasso.create, dict(m=5, n=20, k=2, rho=0.1)),
    Prob("mv_lasso_sparse", lasso.create, dict(m=5, n=20, k=2, rho=0.1, mu=0.1)),
    Prob("qp", qp.create, dict(n=10)),
    Prob("quantile", quantile.create, dict(m=40, n=2, k=3)),
    Prob("robust_pca", robust_pca.create, dict(n=10)),
    Prob("tv_1d", tv_1d.create, dict(n=10)),
]

def solve_problem(problem_instance):
    problem = problem_instance.create()

    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    logging.debug(problem_instance.name)
    params = solver_params_pb2.SolverParams(max_iterations=1000)
    params.rel_tol = REL_TOL.get(problem_instance.name, 1e-3)
    solve.solve(problem, params)
    obj1 = problem.objective.value

    # A lower objective is okay
    assert obj1 <= obj0 + 1e-2*abs(obj0) + 1e-4, "%.2e vs. %.2e" % (obj1, obj0)

def test_solve():
    for problem in PROBLEMS:
        yield solve_problem, problem
