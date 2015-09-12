
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

def get_problems():
    return PROBLEMS

def test_generate_problems():
    for problem in get_test_problems():
        pass
