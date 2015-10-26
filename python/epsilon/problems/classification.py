
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def hinge(x):
    return cp.sum_entries(cp.max_elemwise(x,0))

def create_dense(m, n):
    np.random.seed(0)
    if n < 100:
        p = 1
    else:
        p = 100./n

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])

    x0 = sp.rand(n, 1, p)
    x0.data = np.random.randn(x0.nnz)
    b = np.sign(A*x0 + np.sqrt(0.1)*np.random.randn(m,1))

    return A, b
