
from collections import namedtuple

import numpy as np
import cvxpy as cp

from epsilon import solve

NUM_TRIALS = 10

n = 10
x = cp.Variable(n)
t = cp.Variable(1)

Prox = namedtuple("Prox", ["name", "objective", "constraints"])

PROX_TESTS = [
    Prox("NonNegativeProx", 0, [x >= 0]),
    Prox("NormL1Prox", cp.norm1(x), []),
    Prox("NormL2Prox", cp.norm2(x), []),
    Prox("FusedLassoProx", cp.tv(x), []),
    Prox("NegativeLogProx", -cp.sum_entries(cp.log(x)), []),
    Prox("NegativeEntropyProx", -cp.sum_entries(cp.entr(x)), []),
]

EPIGRAPH_TESTS = [
    Prox("NormL1Epigraph", 0, [cp.norm1(x) <= t]),
    Prox("NormL2Epigraph", 0, [cp.norm2(x) <= t]),
    # TODO(mwytock): Figure out why these are failing
    # Prox("NegativeLogEpigraph", 0, [-cp.sum_entries(cp.log(x)) <= t]),
    # Prox("NegativeEntropyEpigraph", 0, [-cp.sum_entries(cp.entr(x)) <= t]),
]

def test_prox():
    def run(prox, i):
        np.random.seed(i)
        v = np.random.randn(n)

        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) + prox.objective),
                   prox.constraints).solve(solver=cp.SCS)
        expected = {var: var.value for var in (x,)}

        solve.prox(
            cp.Problem(cp.Minimize(prox.objective), prox.constraints), {x: v})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-2)

    for prox in PROX_TESTS:
        for i in xrange(NUM_TRIALS):
            yield run, prox, i

def test_epigraph():
    def run(prox, i):
        np.random.seed(i)

        v = np.random.randn(n)
        s = np.random.randn(1)

        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) +
                               0.5*cp.sum_squares(t - s) +
                               prox.objective),
                   prox.constraints).solve()
        expected = {var: var.value for var in (x,t)}

        solve.prox(
            cp.Problem(cp.Minimize(prox.objective), prox.constraints),
            {x: v, t: s})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(t.value, expected[t], rtol=1e-2, atol=1e-4)

    for prox in EPIGRAPH_TESTS:
        for i in xrange(NUM_TRIALS):
            yield run, prox, i

# TODO(mwytock): Convert test_logistic_prox/epigraph to vectors when logistic
# atom is fixed
def test_prox_logistic():
    def run(i):
        np.random.seed(i)

        v = np.random.randn(1)
        x = cp.Variable(1)
        f = cp.log_sum_exp(cp.vstack(0, x))
        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) + f)).solve()
        expected = {var: var.value for var in (x,)}

        solve.prox(cp.Problem(cp.Minimize(f)), {x: v})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_epigraph_logistic():
    def run(i):
        np.random.seed(i)
        v = np.random.randn(1)
        s = np.random.randn(1)

        x = cp.Variable(1)
        t = cp.Variable(1)
        c = [cp.log_sum_exp(cp.vstack(0, x)) <= t]
        cp.Problem(cp.Minimize(0.5*(cp.sum_squares(x - v) +
                                    cp.sum_squares(t - s))), c).solve()
        expected = {var: var.value for var in (x,t)}

        solve.prox(cp.Problem(cp.Minimize(0), c), {x: v, t: s})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(t.value, expected[t], rtol=1e-2, atol=1e-4)

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
        expected = {var: var.value for var in (x,)}

        solve.prox(cp.Problem(cp.Minimize(0), c), {x: v})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)

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
        expected = {var: var.value for var in (x,y)}

        solve.prox(cp.Problem(cp.Minimize(0), c), {x: v, y: u})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(y.value, expected[y], rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_linear_equality_multivariate():
    """I(z - (y - (1 - Ax)) == 0)"""
    m = 5
    def run(i):
        np.random.seed(i)

        A = np.random.randn(m, n)
        v = np.random.randn(n)
        u = np.random.randn(m)
        w = np.random.randn(m)

        y = cp.Variable(m)
        z = cp.Variable(m)
        c = [z - (y - (1 - A*x)) == 0]
        cp.Problem(
            cp.Minimize(0.5*(cp.sum_squares(x - v) +
                             cp.sum_squares(y - u) +
                             cp.sum_squares(z - w))), c).solve()
        expected = {var: var.value for var in (x, y, z)}

        solve.prox(cp.Problem(cp.Minimize(0), c), {x: v, y: u, z: w})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(y.value, expected[y], rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(z.value, expected[z], rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_non_negative_scaled():
    """I(alpha*x >= 0)"""
    def run(i):
        np.random.seed(i)
        v = np.random.randn(n)
        alpha = np.random.randn()

        x = cp.Variable(n)
        c = [alpha*x >= 0]
        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v)), c).solve()
        expected = {var: var.value for var in (x,)}

        solve.prox(cp.Problem(cp.Minimize(0), c), {x: v})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i

def test_least_squares():
    """||Ax - b||^2"""
    def run(i, m, n):
        np.random.seed(i)
        v = np.random.randn(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        f = cp.sum_squares(A*x  - b)
        cp.Problem(cp.Minimize(f + 0.5*cp.sum_squares(x - v))).solve()
        expected = {var: var.value for var in (x,)}

        solve.prox(cp.Problem(cp.Minimize(f)), {x: v})
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-3)

    for i in xrange(NUM_TRIALS):
        yield run, i, 5, 10

    for i in xrange(NUM_TRIALS):
        yield run, i, 15, 10


def test_least_squares_matrix():
    """||AX - B||_F^2"""
    def run(i, m, n, k):
        np.random.seed(i)
        V = np.random.randn(n, k)
        A = np.random.randn(m, n)
        B = np.random.randn(m, k)

        X = cp.Variable(n, k)
        f = cp.sum_squares(A*X  - B)
        cp.Problem(cp.Minimize(f + 0.5*cp.sum_squares(X - V))).solve()
        expected = {var: var.value for var in (X,)}

        solve.prox(cp.Problem(cp.Minimize(f)), {X: V})
        np.testing.assert_allclose(X.value, expected[X], rtol=1e-2, atol=1e-4)

    for i in xrange(NUM_TRIALS):
        yield run, i, 5, 10, 3

    for i in xrange(NUM_TRIALS):
        yield run, i, 15, 10, 3
