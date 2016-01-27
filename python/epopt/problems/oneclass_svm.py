
from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    m = kwargs["m"]
    n = kwargs["n"]
    lam = 1e3

    x = cp.Variable(A.shape[1])
    rho = cp.Variable(1)
    t = cp.Variable(1)
    f = cp.sum_entries(cp.max_elemwise(
        np.linalg.norm(A, axis=1)**2-2*A*x+t-rho ,0)) + lam * rho
    return cp.Problem(cp.Minimize(f), [rho >= 0, cp.sum_squares(x) <= t])
