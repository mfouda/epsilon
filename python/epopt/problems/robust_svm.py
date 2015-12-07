
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from epopt.problems import problem_util

def create(m, n):
    # Generate data
    X = np.hstack([np.random.randn(m,n), np.ones((m,1))])
    theta0 = np.random.randn(n+1)
    y = np.sign(X.dot(theta0) + 0.1*np.random.randn(m))
    X[y>0,:] += np.tile([theta0], (np.sum(y>0),1))
    X[y<0,:] -= np.tile([theta0], (np.sum(y<0),1))

    # Generate uncertainty envelope
    P = la.block_diag(np.random.randn(n,n), 0)
    lam = 1e-8
    theta = cp.Variable(n+1)

    # TODO(mwytock): write this as:
    # f = (lam/2*cp.sum_squares(theta) +
    #      problem_util.hinge(1 - y[:,np.newaxis]*X*theta+cp.norm1(P.T*theta)))

    # already in prox form
    t1 = cp.Variable(m)
    t2 = cp.Variable(1)
    z = cp.Variable(n+1)
    f = lam/2*cp.sum_squares(theta) + problem_util.hinge(1-t1)
    C = [t1 == y[:,np.newaxis]*X*theta - t2,
         cp.norm1(z) <= t2,
         P.T*theta == z]
    return cp.Problem(cp.Minimize(f), C)
