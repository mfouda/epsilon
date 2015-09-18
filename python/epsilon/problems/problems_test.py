
import cvxpy as cp
import numpy as np

from epsilon.problems import basis_pursuit
from epsilon.problems import covsel
from epsilon.problems import group_lasso
from epsilon.problems import huber
from epsilon.problems import lasso
from epsilon.problems import logreg_l1
from epsilon.problems import lp
from epsilon.problems import tv_1d
from epsilon.problems import tv_smooth

from epsilon.problems.problem_instance import ProblemInstance

# TODO(mwytock): gives wrong answer, likely has to do with linearized ADMM?
# ProblemInstance("tv_1d", tv_1d.create, dict(n=10)),
#
# TODO(mwytock): Need to extend prox_admm.cc to accept more than equality
# constraint. Also, we should be smart about index(variable) nodes.
# ProblemInstance(
#    "group_lasso", group_lasso.create, dict(m=15, ni_max=5, K=10))
#
# TODO(mwytock): Huber prox (or cone reduction) not implemented
# ProblemInstance("huber", huber.create, dict(m=20, n=10))
#
# TODO(mwytock): Logistic prox (or cone reduction) not implemented. Also need to
# support more than one equality constraint
# ProblemInstance("logreg_l1", logreg_l1.create, dict(m=5, n=10))

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    ProblemInstance("lasso", lasso.create, dict(m=5, n=10)),
    ProblemInstance("lp", lp.create, dict(m=10, n=20)),
    ProblemInstance("tv_smooth", tv_smooth.create, dict(n=10, lam=1)),
]

def get_problems():
    return PROBLEMS
