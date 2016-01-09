
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

# Classification
def hinge_loss(theta, X, y):
    return cp.sum_entries(cp.max_elemwise(1 - sp.diags([y],[0])*X*theta, 0))

def logistic_loss(theta, X, y):
    return cp.sum_entries(cp.logistic(-sp.diags([y],[0])*X*theta))

# Multiclass classification
def one_hot(y, k):
    m = len(y)
    return sp.coo_matrix((np.ones(m), (np.arange(m), y)), shape=(m, k)).todense()

def softmax_loss(Theta, X, y):
    k = Theta.size[1]
    return (cp.sum_entries(cp.log_sum_exp(X*Theta, axis=1)) -
            cp.sum_entries(cp.mul_elemwise(one_hot(y, k), X*Theta)))

def multiclass_hinge_loss(Theta, X, y):
    n, k = Theta.size
    ones = np.ones((k,k))
    Y = one_hot(y, k)
    Z = X*Theta
    ZY = cp.mul_elemwise(Y, Z)*ones
    return cp.sum_entries(cp.max_entries(Z - ZY + (1-Y), axis=1))

# Other probabilistic models
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


def poisson_loss(theta, X, y):
    return (cp.sum_entries(cp.exp(X*theta)) -
            cp.sum_entries(sp.diags([y],[0])*X*theta))
