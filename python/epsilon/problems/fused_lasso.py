
import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epsilon.problems import problem_util

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    lam1 = 0.01
    lam2 = 0.01

    x = cp.Variable(A.shape[1])
    f = cp.sum_squares(A*x - b) + lam1*cp.norm1(x) + lam2*cp.tv(x)

    return cp.Problem(cp.Minimize(f))
