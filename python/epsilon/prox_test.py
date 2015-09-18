
import numpy as np
import cvxpy as cp

from epsilon.solve import prox

NUM_TRIALS = 10

def _test_linear_equality(i, m, n):
    np.random.seed(i)
    A = np.random.randn(m,n)
    b = A.dot(np.random.randn(n))
    v = np.random.randn(n)

    x = cp.Variable(n)
    c = [A*x == b]
    cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v)), c).solve()

    x0 = np.asarray(x.value).ravel()
    x1 = prox(cp.Problem(cp.Minimize(0), c), v)
    np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

def test_prox():
    for i in xrange(NUM_TRIALS):
        yield _test_linear_equality, i, 5, 10
