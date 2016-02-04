import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util

def create(**kwargs):
    m = 20
    n = 20
    A = np.random.rand(m,n)
    A -= np.mean(A, axis=0)

    sigma = cp.Variable(n,n)
    t = cp.Variable(1)
    tdet = cp.Variable(1)
    f = t
    C = []
    for i in range(m):
        C.append(cp.quad_form(A[i], sigma) <= t-tdet)
        C.append(-cp.log_det(sigma) <= tdet)
    return cp.Problem(cp.Minimize(f), C)
