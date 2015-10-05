
from collections import namedtuple

import numpy as np
import cvxpy as cp
from numpy.random import randn, rand

from epsilon import solve

NUM_TRIALS = 1

n = 10
x = cp.Variable(n)
t = cp.Variable(1)

class Prox(namedtuple("Prox", ["name", "objective", "constraint"])):
    def __new__(cls, name, objective, constraint=None):
        return super(Prox, cls).__new__(cls, name, objective, constraint)

def f_scaled_zone_single_max():
    alpha = 2*rand()-1
    y = cp.mul_elemwise(randn(n), x) + randn(n)
    return cp.sum_entries(cp.max_elemwise(-alpha*y, (1-alpha)*y))

def f_norm_l1_asymmetric():
    alpha = rand()
    return cp.sum_entries(alpha*cp.max_elemwise(x,0) +
                          (1-alpha)*cp.max_elemwise(-x,0))

def f_dead_zone():
    C = randn()
    return cp.sum_entries(cp.max_elemwise(x-C,0) + cp.max_elemwise(-x-C,0))

def f_hinge():
    return cp.sum_entries(cp.max_elemwise(1-x, 0))

PROX_TESTS = [
    Prox("DeadZoneProx", f_dead_zone),
    Prox("FusedLassoProx", lambda: cp.tv(x)),
    Prox("HingeProx", lambda: cp.sum_entries(cp.max_elemwise(1-x, 0))),
    Prox("LinearProx", lambda: randn(n).T*x),
    Prox("LogisticProx", lambda: cp.sum_entries(cp.logistic(x))),
    Prox("NegativeEntropyProx", lambda: -cp.sum_entries(cp.entr(x))),
    Prox("NegativeLogProx", lambda: -cp.sum_entries(cp.log(x))),
    Prox("NonNegativeProx", None, lambda: x >= 0),
    Prox("NormL1AsymmetricProx", f_norm_l1_asymmetric),
    Prox("NormL1Prox", lambda: cp.norm1(x)),
    Prox("NormL2Prox", lambda: cp.norm2(x)),
    Prox("ScaledZoneProx", f_scaled_zone_single_max),
]

EPIGRAPH_TESTS = [
    Prox("DeadZoneEpigraph", None, lambda: f_dead_zone() <= t),
    Prox("HingeEpigraph", None, lambda: f_hinge() <= t),
    Prox("LogisticEpigraph", None, lambda: cp.sum_entries(cp.logistic(x)) <= t),
    Prox("NormL1AsymmetricEpigraph", None, lambda: f_norm_l1_asymmetric() <= t),
    Prox("NormL1Epigraph", None, lambda: cp.norm1(x) <= t),
    Prox("NormL2Epigraph", None, lambda: cp.norm2(x) <= t),
    # TODO(mwytock): Figure out why these are failing
    # Prox("NegativeLogEpigraph", 0, [-cp.sum_entries(cp.log(x)) <= t]),
    # Prox("NegativeEntropyEpigraph", 0, [-cp.sum_entries(cp.entr(x)) <= t]),
]

def test_prox():
    def run(prox, i):
        np.random.seed(i)
        v = np.random.randn(n)
        lam = np.abs(np.random.randn())

        f = 0 if not prox.objective else prox.objective()
        c = [] if not prox.constraint else [prox.constraint()]
        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) + lam*f), c).solve()
        expected = {var: var.value for var in (x,)}

        solve.prox(cp.Problem(cp.Minimize(f), c), {x: v}, lam)
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-2)

    for prox in PROX_TESTS:
        for i in xrange(NUM_TRIALS):
            yield run, prox, i

def test_epigraph():
    def run(prox, i):
        np.random.seed(i)

        v = np.random.randn(n)
        s = np.random.randn(1)
        lam = np.abs(np.random.randn())

        f = 0 if not prox.objective else prox.objective()
        c = [] if not prox.constraint else [prox.constraint()]
        cp.Problem(cp.Minimize(0.5*cp.sum_squares(x - v) +
                               0.5*cp.sum_squares(t - s) +
                               lam*f), c).solve()
        expected = {var: var.value for var in (x,t)}

        solve.prox(cp.Problem(cp.Minimize(f), c),  {x: v, t: s}, lam)
        np.testing.assert_allclose(x.value, expected[x], rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(t.value, expected[t], rtol=1e-2, atol=1e-4)

    for prox in EPIGRAPH_TESTS:
        for i in xrange(NUM_TRIALS):
            yield run, prox, i

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

def test_linear_equality_multivariate2():
    """I(z - (y - alpha*(A*x - b)))"""
    m = 5
    alpha = 1
    def run(i):
        np.random.seed(i)

        A = np.random.randn(m, n)
        b = np.random.randn(m)
        alpha = np.random.randn()

        v = np.random.randn(n)
        u = np.random.randn(m)
        w = np.random.randn(m)

        y = cp.Variable(m)
        z = cp.Variable(m)
        c = [z - (y - alpha*(A*x - b)) == 0]
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
