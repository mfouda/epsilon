
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epsilon.problems.classification import create_dense, hinge

def create(m, n, k):
    A, b = create_dense(m, n)
    lam = 0.1*sqrt(n)

    P = np.random.randn(k, n)
    x = cp.Variable(n)
    y_p = sp.diags([b.ravel()], [0])*A*x
    f = hinge(1-y_p + cp.norm2(P*w))
    return cp.Problem(cp.Minimize(f))
