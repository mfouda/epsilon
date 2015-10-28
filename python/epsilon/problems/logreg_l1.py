
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epsilon.problems import problem_util

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    lam = 1

    x = cp.Variable(kwargs["n"])
    f = cp.sum_entries(cp.logistic(-sp.diags([b],[0])*A*x)) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
