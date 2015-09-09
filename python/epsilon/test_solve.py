

import unittest

import cvxpy as cp
import numpy as np

from epsilon import solve

class TestSolve(unittest.TestCase):
    def test_lasso(self):
        m = 5
        n = 10

        np.random.seed(0)
        A = np.random.randn(m,n)
        b = np.random.randn(m,1)
        x = cp.Variable(n,1)
        lam = 1

        f = 0.5*cp.sum_squares(A*x - b) + lam*cp.norm1(x);
        status = solve.solve(cp.Problem(cp.Minimize(f)))
