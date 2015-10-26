"""Standard SVM, i.e.. hinge loss w/ l2 regularization."""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epsilon.problems.classification import create_dense, hinge

def create(m, n):
    A, b = create_dense(m, n)
    lam = 0.1*np.sqrt(n)

    x = cp.Variable(n)
    y_p = sp.diags([b.ravel()], [0])*A*x
    f = hinge(1-y_p) + lam*cp.sum_squares(x)
    return cp.Problem(cp.Minimize(f))
