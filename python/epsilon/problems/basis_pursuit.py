
import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def create(m, n):
    A = np.random.randn(m,n)
    x0 = sp.rand(n, 1, 0.1)
    b = A*x0

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.norm1(x)), [A*x == b])
