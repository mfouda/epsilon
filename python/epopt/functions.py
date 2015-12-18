
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
    Z = X*Theta - Y
    return cp.sum_entries(
        cp.max_elemwise(
            cp.mul_elemwise( -A, Z),
            cp.mul_elemwise(1-A, Z)))

def softmax_loss(Theta, X, y):
    m = len(y)
    n, k = Theta.size
    Y = sp.coo_matrix((np.ones(m), (np.arange(m), y)), shape=(m, k))
    print cp.__file__
    return (cp.sum_entries(cp.log_sum_exp(X*Theta, axis=1)) -
            cp.sum_entries(cp.mul_elemwise(Y, X*Theta)))

def poisson_loss(theta, X, y):
    return (cp.sum_entries(cp.exp(X*theta)) -
            cp.sum_entries(sp.diags([y],[0])*X*theta))
