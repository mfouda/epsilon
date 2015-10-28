
from collections import namedtuple

import numpy as np
import cvxpy as cp
from numpy.random import randn, rand

from epsilon import solve

PROX_TRIALS = 10

# Common variable
n = 10
x = cp.Variable(n)
p = cp.Variable(3)
X = cp.Variable(3,3)
t = cp.Variable(1)
p1 = cp.Variable(1)
q1 = cp.Variable(1)

class Prox(namedtuple("Prox", ["name", "objective", "constraint"])):
    def __new__(cls, name, objective, constraint=None):
        return super(Prox, cls).__new__(cls, name, objective, constraint)

def f_scaled_zone_single_max():
    alpha = rand()
    y = cp.mul_elemwise(randn(n), x) + randn(n)
    return cp.sum_entries(cp.max_elemwise(-alpha*y, (1-alpha)*y))

def f_norm_l1_asymmetric():
    alpha = rand()
    return cp.sum_entries(alpha*cp.max_elemwise(x,0) +
                          (1-alpha)*cp.max_elemwise(-x,0))

def f_dead_zone():
    return cp.sum_entries(cp.max_elemwise(x-1,0) + cp.max_elemwise(-x-1,0))

def f_hinge():
    return cp.sum_entries(cp.max_elemwise(1-x, 0))

def f_least_squares(m):
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    return cp.sum_squares(A*x - b)

def f_least_squares_matrix():
    m = 20
    k = 3
    A = np.random.randn(m, n)
    B = np.random.randn(m, k)
    X = cp.Variable(n, k)
    return cp.sum_squares(A*X  - B)

def C_linear_equality():
    m = 5
    A = np.random.randn(m, n)
    b = A.dot(np.random.randn(n))
    return [A*x == b]

def C_linear_equality_matrix_lhs():
    m = 5
    k = 3
    A = np.random.randn(m, n)
    X = cp.Variable(n, k)
    B = A.dot(np.random.randn(n, k))
    return [A*X == B]

def C_linear_equality_matrix_rhs():
    m = 3
    k = 5
    A = np.random.randn(k, m)
    X = cp.Variable(n, k)
    B = np.random.randn(n, k).dot(A)
    return [X*A == B]

def C_linear_equality_graph(m):
    A = np.random.randn(m, n)
    y = cp.Variable(m)
    return [y == A*x]

def C_linear_equality_graph_lhs(m, n):
    k = 3
    A = np.random.randn(m, n)
    B = A.dot(np.random.randn(n,k))
    X = cp.Variable(n, k)
    Y = cp.Variable(m, k)
    return [Y == A*X + B]

def C_linear_equality_graph_rhs(m, n):
    k = 3
    A = np.random.randn(m, n)
    B = np.random.randn(k, m).dot(A)
    X = cp.Variable(k, m)
    Y = cp.Variable(k, n)
    return [Y == X*A + B]

def C_linear_equality_multivariate():
    m = 5
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    alpha = np.random.randn()
    y = cp.Variable(m)
    z = cp.Variable(m)
    return [z - (y - alpha*(A*x - b)) == 0]

def C_linear_equality_multivariate2():
    m = 5
    A = np.random.randn(m, n)
    y = cp.Variable(m)
    z = cp.Variable(m)
    return [z - (y - (1 - A*x)) == 0]

def C_non_negative_scaled():
    alpha = np.random.randn()
    return [alpha*x >= 0]

# Proximal operators
PROX_TESTS = [
    #Prox("MatrixFracProx", lambda: cp.matrix_frac(p, X)),
    Prox("DeadZoneProx", f_dead_zone),
    Prox("FusedLassoProx", lambda: cp.tv(x)),
    Prox("HingeProx", lambda: cp.sum_entries(cp.max_elemwise(1-x, 0))),
    Prox("InvPosProx", lambda: cp.sum_entries(cp.inv_pos(x))),
    Prox("KLDivProx", lambda: cp.kl_div(p1, q1)),
    Prox("LambdaMaxProx", lambda: cp.lambda_max(X)),
    Prox("LeastSquaresProx", lambda: f_least_squares(5)),
    Prox("LeastSquaresProx", lambda: f_least_squares(20)),
    Prox("LeastSquaresProx", f_least_squares_matrix),
    Prox("LinearEqualityProx", None, C_linear_equality),
    Prox("LinearEqualityProx", None, C_linear_equality_matrix_lhs),
    Prox("LinearEqualityProx", None, C_linear_equality_matrix_rhs),
    Prox("LinearEqualityProx", None, C_linear_equality_multivariate),
    Prox("LinearEqualityProx", None, C_linear_equality_multivariate2),
    Prox("LinearEqualityProx", None, lambda: C_linear_equality_graph(20)),
    Prox("LinearEqualityProx", None, lambda: C_linear_equality_graph(5)),
    Prox("LinearEqualityProx", None, lambda: C_linear_equality_graph_lhs(10, 5)),
    Prox("LinearEqualityProx", None, lambda: C_linear_equality_graph_lhs(5, 10)),
    Prox("LinearEqualityProx", None, lambda: C_linear_equality_graph_rhs(10, 5)),
    Prox("LinearEqualityProx", None, lambda: C_linear_equality_graph_rhs(5, 10)),
    Prox("LinearProx", lambda: randn(n).T*x),
    Prox("LogisticProx", lambda: cp.sum_entries(cp.logistic(x))),
    Prox("MaxEntriesProx", lambda: cp.max_entries(x)),
    Prox("NegativeEntropyProx", lambda: -cp.sum_entries(cp.entr(x))),
    Prox("NegativeLogDetProx", lambda: -cp.log_det(X)),
    Prox("NegativeLogProx", lambda: -cp.sum_entries(cp.log(x))),
    Prox("NonNegativeProx", None, C_non_negative_scaled),
    Prox("NonNegativeProx", None, lambda: [x >= 0]),
    Prox("NormFrobeniusProx", lambda: cp.norm(X, "fro")),
    Prox("NormL1AsymmetricProx", f_norm_l1_asymmetric),
    Prox("NormL1Prox", lambda: cp.norm1(x)),
    Prox("NormL2Prox", lambda: cp.norm2(x)),
    Prox("NormNuclearProx", lambda: cp.norm(X, "nuc")),
    Prox("ScaledZoneProx", f_scaled_zone_single_max),
    Prox("SemidefiniteProx", None, lambda: [X >> 0]),
    Prox("SumExpProx", lambda: cp.sum_entries(cp.exp(x))),
    Prox("SumLargest", lambda: cp.sum_largest(x, 4)),
]

# Epigraph operators
PROX_TESTS += [
    Prox("DeadZoneEpigraph", None, lambda: [f_dead_zone() <= t]),
    Prox("HingeEpigraph", None, lambda: [f_hinge() <= t]),
    Prox("InvPosEpigraph", None, lambda: [cp.sum_entries(cp.inv_pos(x)) <= t]),
    Prox("KLDivEpigraph", None, lambda: [cp.kl_div(p1,q1) <= t]),
    Prox("LambdaMaxEpigraph", None, lambda: [cp.lambda_max(X) <= t]),
    Prox("LogisticEpigraph", None, lambda: [cp.sum_entries(cp.logistic(x)) <= t]),
    Prox("MaxEntriesEpigraph", None, lambda: [cp.max_entries(x) <= t]),
    Prox("NegativeEntropyEpigraph", None, lambda: [-cp.sum_entries(cp.entr(x)) <= t]),
    Prox("NegativeLogDetEpigraph", None, lambda: [-cp.log_det(X) <= t]),
    Prox("NegativeLogEpigraph", None, lambda: [-cp.sum_entries(cp.log(x)) <= t]),
    Prox("NormFrobeniusEpigraph", None, lambda: [cp.norm(X, "fro") <= t]),
    Prox("NormL1AsymmetricEpigraph", None, lambda: [f_norm_l1_asymmetric() <= t]),
    Prox("NormL1Epigraph", None, lambda: [cp.norm1(x) <= t]),
    Prox("NormL2Epigraph", None, lambda: [cp.norm2(x) <= t]),
    Prox("NormNuclearEpigraph", None, lambda: [cp.norm(X, "nuc") <= t]),
    Prox("SumExpEpigraph", None, lambda: [cp.sum_entries(cp.exp(x)) <= t]),
]

def test_prox():
    def run(prox, i):
        np.random.seed(i)
        v = np.random.randn(n)
        lam = np.abs(np.random.randn())

        f = 0 if not prox.objective else prox.objective()
        C = [] if not prox.constraint else prox.constraint()

        # Form problem and solve with proximal operator implementation
        prob = cp.Problem(cp.Minimize(f), C)
        v_map = {x: np.random.randn(*x.size) for x in prob.variables()}
        #for (k,v) in v_map.items():
        #    if v.shape[0] == v.shape[1]:
        #        v_map[k] = v.dot(v.T) #(v + v.T)/2
        solve.prox(prob, v_map, lam)
        actual = {x: x.value for x in prob.variables()}

        # Compare to solution with cvxpy
        prob.objective.args[0] *= lam
        prob.objective.args[0] += sum(
            0.5*cp.sum_squares(x - v_map[x]) for x, v in v_map.iteritems())
        prob.solve()

        try:
            for x in prob.variables():
                np.testing.assert_allclose(x.value, actual[x], rtol=1e-2, atol=1e-2)
        except AssertionError as e:
            # print objective value and constraints
            print
            print 'cvx:'
            print map(lambda x: x.value, prob.variables())
            print 'actual:'
            print actual.values()
            print 'vmap:'
            print v_map.values()
            print 'cvx obj:', prob.objective.value
            for c in prob.constraints:
                print c, c.value, map(lambda x: x.value, c.args)

            for x,v in actual.items():
                x.value = v
            print 'our obj:', prob.objective.value
            for c in prob.constraints:
                print c, c.value, map(lambda x: x.value, c.args)
            print

            raise e

    for prox in PROX_TESTS:
        for i in xrange(PROX_TRIALS):
            yield run, prox, i
