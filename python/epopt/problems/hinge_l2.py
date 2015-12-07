"""Standard SVM, i.e.. hinge loss w/ l2 regularization."""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epopt.problems import problem_util

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    lam = 1

    x = cp.Variable(A.shape[1])
    f = problem_util.hinge(1-sp.diags([b],[0])*A*x) + lam*cp.sum_squares(x)
    return cp.Problem(cp.Minimize(f))
