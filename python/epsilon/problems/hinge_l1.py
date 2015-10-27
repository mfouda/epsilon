
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from epsilon.problems.classification import hinge, create_dense

def create(m, n):
    A, b = create_dense(m, n)
    lam = 0.1*np.sqrt(n)

    x = cp.Variable(n)
    y_p = sp.diags([b.ravel()], [0])*A*x
    f = hinge(1-y_p) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
