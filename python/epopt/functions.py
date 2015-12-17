
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def hinge_loss(theta, X, y):
    return cp.sum_entries(cp.max_elemwise(1 - sp.diags([y],[0])*X*theta, 0))

def logistic_loss(theta, X, y):
    return cp.sum_entries(cp.logistic(-sp.diags([y],[0])*X*theta))

def quantile_loss(alphas, Theta, X, y):
    m, n = X.shape
    k = len(alphas)
    Y = np.tile(y, (k, 1)).T
    A = np.tile(alphas, (m, 1))
    XT = X*Theta
    return cp.sum_entries(
        cp.max_elemwise(
            cp.mul_elemwise( -A, XT - Y),
            cp.mul_elemwise(1-A, XT - Y)))
