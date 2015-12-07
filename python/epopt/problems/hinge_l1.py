
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epopt.problems import problem_util

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    m = kwargs["m"]
    n = kwargs["n"]
    sigma = 0.05
    mu = kwargs.get("mu", 1)
    lam = 0.5*sigma*np.sqrt(m*np.log(mu*n))

    x = cp.Variable(A.shape[1])
    f =  problem_util.hinge(1-sp.diags([b],[0])*A*x) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
