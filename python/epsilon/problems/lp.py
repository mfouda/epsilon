
import numpy as np
import cvxpy as cp

def create(m, n):
    A = np.abs(np.random.randn(m,n))
    b = A.dot(np.random.randn(n))
    c = np.random.rand(n) + 0.5

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(c.T*x), [A*x == b, x >= 0])
