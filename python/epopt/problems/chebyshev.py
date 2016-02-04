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
    p = 2

    x = cp.Variable(n)
    t1 = cp.Variable(k)
    t = cp.Variable(k)
    v = cp.Variable(k)
    f = cp.sum_largest(t, p)
    C = []
    for i in range(k):
        C += [cp.pnorm(A[i]*x, 1) <= t1[i]]
    C += [v == B*x+c, v <= (t-t1), -v <= (t-t1)]

    f_eval = lambda: cp.sum_largest(np.array([cp.pnorm(A[i]*x, 1).value for i in range(k)]) + (B*x+c).value, p).value

    return cp.Problem(cp.Minimize(f), C), f_eval
