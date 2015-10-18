
import logging

import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.problems import basis_pursuit
from epsilon.problems import covsel
from epsilon.problems import group_lasso
from epsilon.problems import hinge_l1
from epsilon.problems import huber
from epsilon.problems import lasso
from epsilon.problems import least_abs_dev
from epsilon.problems import logreg_l1
from epsilon.problems import lp
from epsilon.problems import mnist
from epsilon.problems import quantile
from epsilon.problems import tv_1d
from epsilon.problems import tv_denoise
from epsilon.problems.problem_instance import ProblemInstance

# Override accuracy settings
REL_TOL = {}

# Needs HSTACK implementation and linearization
# ProblemInstance("tv_denoise", tv_denoise.create, dict(n=10, lam=1)),

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    ProblemInstance("group_lasso", group_lasso.create, dict(m=15, ni=5, K=10)),
    ProblemInstance("hinge_l1", hinge_l1.create, dict(m=5, n=10)),
    ProblemInstance("huber", huber.create, dict(m=20, n=10)),
    ProblemInstance("lasso", lasso.create, dict(m=5, n=10)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=10, n=5)),
    ProblemInstance("logreg_l1", logreg_l1.create, dict(m=5, n=10)),
    ProblemInstance("lp", lp.create, dict(m=10, n=20)),
    ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_TINY, n=10)),
    ProblemInstance("quantile", quantile.create, dict(m=40, n=2, k=3)),
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
