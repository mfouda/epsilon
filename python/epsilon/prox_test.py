
import numpy as np
import cvxpy as cp

from epsilon.solve import prox

NUM_TRIALS = 10

def _test_linear_equality_simple(i, m, n):
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

def _test_linear_equality_graph(i, m, n):
    np.random.seed(i)

    A = np.random.randn(m,n)
    v = np.random.randn(n)
    u = np.random.randn(m)

    x = cp.Variable(n)
    y = cp.Variable(m)
    c = [A*x == y]
    cp.Problem(
        cp.Minimize(0.5*(cp.sum_squares(x - v) +
                         cp.sum_squares(y - u))),
        c).solve()
    x0 = np.asarray(x.value).ravel()
    y0 = np.asarray(y.value).ravel()

    xy = prox(cp.Problem(cp.Minimize(0), c), np.hstack((v, u)))
    np.testing.assert_allclose(np.hstack((x0, y0)), xy, rtol=1e-2, atol=1e-4)

def _test_non_negative_simple(i, n):
    np.random.seed(i)
    v = np.random.randn(n)

    x = cp.Variable(n)
    c = [x >= 0]
    cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v)), c).solve()

    x0 = np.asarray(x.value).ravel()
    x1 = prox(cp.Problem(cp.Minimize(0), c), v)
    np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

def _test_non_negative_scaled(i, n):
    np.random.seed(i)
    b = np.random.randn(n)
    v = np.random.randn(n)
    alpha = np.random.randn()

    x = cp.Variable(n)
    c = [alpha*x + b >= 0]
    cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v)), c).solve()

    x0 = np.asarray(x.value).ravel()
    x1 = prox(cp.Problem(cp.Minimize(0), c), v)
    np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

def _test_norm2_simple(i, n):
    np.random.seed(i)
    v = np.random.randn(n)

    x = cp.Variable(n)
    f = cp.norm2(x)
    cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) + f)).solve()

    x0 = np.asarray(x.value).ravel()
    x1 = prox(cp.Problem(cp.Minimize(f)), v)
    np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

def _test_norm2_epigraph(i, n):
    np.random.seed(i)
    v = np.random.randn(n)
    s = np.random.randn()

    x = cp.Variable(n)
    t = cp.Variable(1)
    c = [cp.norm2(x) <= t]
    cp.Problem(cp.Minimize(0.5*(cp.sum_squares(x - v) +
                                cp.sum_squares(t - s))), c).solve()

    xt0 = np.asarray(np.vstack((t.value, x.value))).ravel()
    xt1 = prox(cp.Problem(cp.Minimize(0), c), np.hstack((s, v)))
    np.testing.assert_allclose(xt0, xt1, rtol=1e-2, atol=1e-4)

def test_linear_equality():
    for i in xrange(NUM_TRIALS):
        yield _test_linear_equality_simple, i, 5, 10
    for i in xrange(NUM_TRIALS):
        yield _test_linear_equality_graph, i, 5, 10

def test_non_negative():
    for i in xrange(NUM_TRIALS):
        yield _test_non_negative_simple, i, 10
    for i in xrange(NUM_TRIALS):
        yield _test_non_negative_scaled, i, 10

def test_norm2():
    for i in xrange(NUM_TRIALS):
        yield _test_norm2_simple, i, 1
    for i in xrange(NUM_TRIALS):
        yield _test_norm2_simple, i, 10

def test_norm2_epigraph():
    for i in xrange(NUM_TRIALS):
        yield _test_norm2_epigraph, i, 10
