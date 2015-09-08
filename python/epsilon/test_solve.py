

import unittest

import cvxpy as cp
import numpy as np

from epsilon import solve

class TestSolve(unittest.TestCase):
    def test_ls(self):
        m = 10
        n = 5

        np.random.seed(0)
        A = np.random.randn(m,n)
        b = np.random.randn(m,1)
        x = cp.Variable(n,1)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(A*x - b)))
        status = solve.solve(prob)
