
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from epsilon.problems.classification import create_dense, logistic_loss

def create(m, n):
    A, b = create_dense(m, n)
    lam = 0.1*np.sqrt(n)

    x = cp.Variable(n)
    y_p = sp.diags([b.ravel()], [0])*A*x
    f = logistic_loss(y_p) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
