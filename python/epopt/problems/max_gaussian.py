import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util

def create(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    k = kwargs["k"]
    A = np.matrix(np.random.rand(m,n))
    A -= np.mean(A, axis=0)
    K = np.array([(A[i].T*A[i]).flatten() for i in xrange(m)])

    sigma = cp.Variable(n,n)
    t = cp.Variable(m)
    tdet = cp.Variable(1)
    f = cp.sum_largest(t+tdet, k)
    z = K*cp.reshape(sigma, n*n, 1)
    C = [-cp.log_det(sigma) <= tdet, t == z]

    return cp.Problem(cp.Minimize(f), C)
