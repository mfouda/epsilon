
from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    lam = 1e3
    A = problem_util.normalized_data_matrix(n,m,1)

    x = cp.Variable(n)
    rho = cp.Variable(1)
    t = cp.Variable(1)
    f = cp.sum_entries(cp.max_elemwise(
        1-2*A.T*x+t-rho, 0)) + lam * rho
    
    sq_eval = lambda: cp.sum_squares(x).value
    f_eval = lambda: (cp.sum_entries(cp.max_elemwise(1-2*A.T*x+sq_eval()-rho, 0)) + lam*cp.max_elemwise(0,rho)).value

    return cp.Problem(cp.Minimize(f), [rho >= 0, cp.sum_squares(x) <= t]), f_eval
