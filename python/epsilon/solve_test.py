
import logging

import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.problems import *
from epsilon.problems.problem_instance import ProblemInstance

# Override accuracy settings
REL_TOL = {}

# Need convolution operators
# ProblemInstance("tv_denoise", tv_denoise.create, dict(n=10, lam=1)),
#
# Huge expression tree. consider way to do graph problems?
# ProblemInstance("map_inference", map_inference.create, dict(n=10)),
#
# Need to fix ScaledZoneProx family functions to make canonicalize robust
# ProblemInstance("robust_svm", robust_svm.create, dict(m=20, n=10, k=3)),
#
# TODO, sparse examples:
#
# group_lasso_sparse
# hinge_l1_sparse
# hinge_l2_sparse
# lasso_sparse
# logreg_l1_sparse
#
# TODO, convolution examples
#
# conv_1d
# conv_2d

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    ProblemInstance("group_lasso", group_lasso.create, dict(m=15, ni=5, K=10)),
    ProblemInstance("hinge_l1", hinge_l1.create, dict(m=5, n=10)),
    ProblemInstance("hinge_l2", hinge_l2.create, dict(m=20, n=10)),
    ProblemInstance("huber", huber.create, dict(m=20, n=10)),
    ProblemInstance("lasso", lasso.create, dict(m=5, n=10)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=10, n=5)),
    ProblemInstance("logreg_l1", logreg_l1.create, dict(m=5, n=10)),
    ProblemInstance("lp", lp.create, dict(m=10, n=20)),
    ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_TINY, n=10)),
    ProblemInstance("portfolio", portfolio.create, dict(m=5, n=10)),
    ProblemInstance("qp", qp.create, dict(n=10)),
    ProblemInstance("quantile", quantile.create, dict(m=40, n=2, k=3)),
    ProblemInstance("robust_pca", robust_pca.create, dict(n=10)),
    ProblemInstance("tv_1d", tv_1d.create, dict(n=10)),
]

def solve_problem(problem_instance):
    problem = problem_instance.create()

    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    logging.debug(problem_instance.name)
    params = solver_params_pb2.SolverParams(max_iterations=1000)
    params.rel_tol = REL_TOL.get(problem_instance.name, 1e-2)
    solve.solve(problem, params)
    obj1 = problem.objective.value

    # A lower objective is okay
    assert obj1 <= obj0 + 1e-2*abs(obj0) + 1e-4, "%.2e vs. %.2e" % (obj1, obj0)

def test_solve():
    for problem in PROBLEMS:
        yield solve_problem, problem
