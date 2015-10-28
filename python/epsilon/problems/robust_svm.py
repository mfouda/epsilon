
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epsilon.problems import problem_util

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    lam = 1

    P = np.random.randn(k, n)
    x = cp.Variable(A.shape[1])
    f = (problem_util.hinge(1 - sp.diags([b],[0])*A*x + cp.norm2(P*x)) +
         lam*cp.sum_squares(x))
    return cp.Problem(cp.Minimize(f))
