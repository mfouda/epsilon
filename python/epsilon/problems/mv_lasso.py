
import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp

def create(m, n, k):
    if n < 100:
        p = 1
    else:
        p = 100./n

    np.random.seed(0)
    A = np.random.randn(m,n);
    A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])

    X0 = sp.rand(n,k,p);
    X0.data = np.random.randn(X0.nnz)
    B = A*X0 + np.sqrt(0.001)*np.random.randn(m,k)

    lambda_max = np.max(np.abs(A))
    lam = 0.1*lambda_max

    X = cp.Variable(n,k)
    return cp.Problem(cp.Minimize(cp.sum_squares(A*X - B) + lam*cp.norm(X,1)))
