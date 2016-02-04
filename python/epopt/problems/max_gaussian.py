import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util

def create(**kwargs):
    m = 2
    n = 20
    p = 1
    A = np.random.rand(m,n)
    A -= np.mean(A, axis=0)

    sigma = cp.Variable(n,n)
    t = cp.Variable(m)
    tdet = [cp.Variable(1) for i in range(m)]
    f = cp.sum_largest(t, p)
    C = []
    for i in range(m):
        C.append(-cp.log_det(sigma) <= tdet[i])
        C.append(cp.quad_form(A[i], sigma) <= t[i]-tdet[i])
    return cp.Problem(cp.Minimize(f), C)
