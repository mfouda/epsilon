
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from epsilon.problems import classification

def logistic(x):
    return sum(cp.log_sum_exp(cp.vstack(0, -x[i])) for i in xrange(x.size[0]))

def create(m, n):
    A, b = classification.create_dense(m, n)
    lam = 0.1*np.sqrt(n)

    x = cp.Variable(n)
    y_p = sp.diags([b.ravel()], [0])*A*x
    f = logistic_loss(y_p) + lam*cp.norm1(x)
    return cp.Problem(cp.Minimize(f))
