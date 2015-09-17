
import cvxpy as cp
import numpy as np

from epsilon.problems import basis_pursuit
from epsilon.problems import covsel
from epsilon.problems import lasso
from epsilon.problems import tv_smooth
from epsilon.problems.problem_instance import ProblemInstance

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    # ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    # ProblemInstance("lasso", lasso.create, dict(m=5, n=10)),
    # ProblemInstance("tv_smooth", tv_smooth.create, dict(n=10, lam=1)),
]

def get_problems():
    return PROBLEMS
