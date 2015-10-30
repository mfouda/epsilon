
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from epsilon.problems import problem_util

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)

    # TODO(mwytock): This is based on lambda_max for logreg_l1, find the right
    # rule here.
    ratio = float(np.sum(b==1)) / len(b)
    lambda_max = np.abs((1-ratio)*A[b==1,:].sum(axis=0) +
                        ratio*A[b==-1,:].sum(axis=0)).max()
    lam = 0.9*lambda_max

    x = cp.Variable(A.shape[1])
    f = problem_util.hinge(1-sp.diags([b],[0])*A*x) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
