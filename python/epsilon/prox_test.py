
from collections import namedtuple

import numpy as np
import cvxpy as cp

from epsilon import solve

NUM_TRIALS = 10

m = 5
n = 10
x = cp.Variable(n)
t = cp.Variable(1)

Prox = namedtuple("Prox", ["name", "objective", "constraints"])

PROX_TESTS = [
    Prox("NonNegativeProx", 0, [x >= 0]),
    Prox("NormL1Prox", cp.norm1(x), []),
    Prox("NormL2Prox", cp.norm2(x), []),
    Prox("FusedLassoProx", cp.tv(x), []),
    # TODO(mwytock): Need project feasible methods
    # Prox("NegativeLogProx", -cp.sum_entries(cp.log(x)), []),
    # Prox("NegativeEntropyProx", -cp.sum_entries(cp.entr(x)), []),
]

EPIGRAPH_TESTS = [
    Prox("NormL1Epigraph", 0, [cp.norm1(x) <= t]),
    Prox("NormL2Epigraph", 0, [cp.norm2(x) <= t]),
    # Prox("NegativeLogEpigraph", 0, [-cp.sum_entries(cp.log(x)) <= t]),
    # Prox("NegativeEntropyEpigraph", 0, [-cp.sum_entries(cp.entr(x)) <= t]),
]

def test_prox():
    def run(prox, i):
        np.random.seed(i)
        v = np.random.randn(n)

        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) + prox.objective),
                   prox.constraints).solve()
        x0 = np.asarray(x.value).ravel()
        x1 = solve.prox(cp.Problem(cp.Minimize(prox.objective), prox.constraints), v)
        np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

    for prox in PROX_TESTS:
        for i in xrange(NUM_TRIALS):
            yield run, prox, i

def test_epigraph():
    def run(prox, i):
        np.random.seed(i)

        v = np.random.randn(n)
        s = np.random.randn()

        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) +
                               0.5*cp.sum_squares(t - s) +
                               prox.objective),
                   prox.constraints).solve()

        tx0 = np.asarray(np.vstack((t.value, x.value))).ravel()
        tx1 = solve.prox(
            cp.Problem(cp.Minimize(prox.objective), prox.constraints),
            np.hstack((s, v)))
        np.testing.assert_allclose(tx0, tx1, rtol=1e-2, atol=1e-4)

    for prox in EPIGRAPH_TESTS:
        for i in xrange(NUM_TRIALS):
            yield run, prox, i

# TODO(mwytock): Convert test_logistic_prox/epigraph to vectors when logistic
# atom is fixed
def test_logistic_prox():
    def run(i):
        np.random.seed(i)

        v = np.random.randn(1)
        x = cp.Variable(1)
        f = cp.log_sum_exp(cp.vstack(0, x))
        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) + f)).solve()

        x0 = np.asarray(x.value).ravel()
        x1 = solve.prox(cp.Problem(cp.Minimize(f)), v)
        np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_logistic_epigraph():
    def run(i):
        np.random.seed(i)
        v = np.random.randn(1)
        s = np.random.randn()

        x = cp.Variable(1)
        t = cp.Variable(1)
        c = [cp.log_sum_exp(cp.vstack(0, x)) <= t]
        cp.Problem(cp.Minimize(0.5*(cp.sum_squares(x - v) +
                                    cp.sum_squares(t - s))), c).solve()

        tx0 = np.asarray(np.vstack((t.value, x.value))).ravel()
        tx1 = solve.prox(cp.Problem(cp.Minimize(0), c), np.hstack((s, v)))
        np.testing.assert_allclose(tx0, tx1, rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_linear_equality():
    """I(Ax == b)"""
    m = 5
    def run(i):
        np.random.seed(i)
        A = np.random.randn(m, n)
        b = A.dot(np.random.randn(n))
        v = np.random.randn(n)

        c = [A*x == b]
        cp.Problem(cp.Minimize(0.5*(cp.sum_squares(x - v))), c).solve()
        x0 = np.asarray(x.value).ravel()
        x1 = solve.prox(cp.Problem(cp.Minimize(0), c), v)
        np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_linear_equality_graph():
    """I(Ax == y)"""
    m = 5
    def run(i):
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

        xy0 = np.asarray(np.vstack((x.value, y.value))).ravel()
        xy1 = solve.prox(cp.Problem(cp.Minimize(0), c), np.hstack((v, u)))
        np.testing.assert_allclose(xy0, xy1, rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_non_negative_scaled():
    """I(alpha*x >= 0)"""
    def run(i):
        np.random.seed(i)
        v = np.random.randn(n)
        alpha = np.random.randn()

        x = cp.Variable(n)
        c = [alpha*x  >= 0]
        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v)), c).solve()

        x0 = np.asarray(x.value).ravel()
        x1 = solve.prox(cp.Problem(cp.Minimize(0), c), v)
        np.testing.assert_allclose(x0, x1, rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i
