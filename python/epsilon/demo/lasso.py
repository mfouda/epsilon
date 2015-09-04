
import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp

def create(m, n):
    m = int(m)
    n = int(n)
    if n < 100:
        p = 1
    else:
        p = 100./n

    np.random.seed(0)
    A = np.random.randn(m,n);
    A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])

    x0 = sp.rand(n,1,p);
    x0.data = np.random.randn(x0.nnz)
    b = A*x0 + np.sqrt(0.001)*np.random.randn(m,1)

    lambda_max = LA.norm(A.T.dot(b), np.inf)
    lam = 0.1*lambda_max

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_squares(A*x - b) + lam*cp.norm(x,1)))
