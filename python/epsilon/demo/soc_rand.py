
import cvxpy as cp
import numpy as np

def create(m, n):
    np.random.rand(0)
    A = np.random.rand(m, n)
    b = np.random.rand(m, 1)
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.norm(A*x - b)))
