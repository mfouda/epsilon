
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from epsilon.problems import classification

def hinge(x):
    return cp.sum_entries(cp.max_elemwise(0, 1-x))

def create(m, n):
    A, b = classification.create_dense(m, n)
    lam = 0.1*np.sqrt(n)

    x = cp.Variable(n)
    y_p = sp.diags([b.ravel()], [0])*A*x
    f = hinge(y_p) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
