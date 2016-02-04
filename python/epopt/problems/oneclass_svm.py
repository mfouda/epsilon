
from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(m, n):
    # Generate random points uniform over hypersphere
    A = np.random.randn(m, n)
    A /= np.sqrt(np.sum(A**2, axis=1))[:,np.newaxis]
    A *= (np.random.rand(m)**(1./n))[:,np.newaxis]

    # Shift points and add some outliers
    x0 = np.random.randn(n)
    A += x0
    k = max(m/50, 1)
    idx = np.random.randint(0, m, k)
    A[idx, :] += np.random.randn(k, n)

    lam = np.sqrt(m)
    x = cp.Variable(n)
    rho = cp.Variable(1)

    z = np.sum(A**2, axis=1) - 2*A*x + cp.sum_squares(x)  # z_i = ||a_i - x||^2
    f = (lam*cp.max_elemwise(0, rho) + cp.sum_entries(cp.max_elemwise(z - rho, 0)))
    return cp.Problem(cp.Minimize(f))
