from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    k = kwargs["k"]
    A = [problem_util.normalized_data_matrix(m,n,1) for i in range(k)]
    B = problem_util.normalized_data_matrix(k,n,1)
    c = np.random.rand(k)

    x = cp.Variable(n)
    t = cp.Variable(1)
    t1 = cp.Variable(1)
    u = [cp.Variable(m) for i in range(k)]
    v = cp.Variable(k)
    f = t
    C = []
    for i in range(k):
        C += [cp.pnorm(A[i]*x, 1) <= t1]
    C += [v == B*x+c, t1 >= 0, v <= (t-t1)*np.ones(k), -v <= (t-t1)*np.ones(k)]
    return cp.Problem(cp.Minimize(f), C)
