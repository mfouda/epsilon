
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def hinge(x):
    return cp.sum_entries(cp.max_elemwise(x,0))

def create_regression(m, n, k=1, rho=1, mu=1, sigma=0.01):
    """Create a random (multivariate) regression problem."""

    X0 = sp.rand(n, k, rho)
    X0.data = np.random.randn(X0.nnz)

    if mu == 1:
        A = np.random.randn(m, n)
    else:
        A = sp.rand(m, n, mu)
        A.data = np.random.randn(A.nnz)
        B = A.dot(X0) + sigma*np.random.randn(m, k)

    if k == 1:
        x0 = sp.rand(n, 1, rho)
        x0.data = np.random.randn(x0.nnz)
        x0 = x0.toarray().ravel()
        b = A.dot(x0) + sigma*np.random.randn(m)
        return A, b
    else:
        X0 = sp.rand(n, k, rho)
        X0.data = np.random.randn(X0.nnz)
        X0 = X0.toarray()
        B = A.dot(X0) + sigma*np.random.randn(m,k)
        return A, B

def create_classification(m, n, rho=1, mu=1, sigma=0.1):
    """Create a random classification problem."""

    x0 = sp.rand(n, 1, rho)
    x0.data = np.random.randn(x0.nnz)
    x0 = x0.toarray().ravel()

    if mu == 1:
        A = np.random.randn(m, n)
    else:
        A = sp.rand(m, n, mu)
        A.data = np.random.randn(A.nnz)

    b = np.sign(A.dot(x0) + sigma*np.random.randn(m))
    return A, b
