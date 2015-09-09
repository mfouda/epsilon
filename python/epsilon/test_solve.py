
import cvxpy as cp
import numpy as np

from epsilon import solve
from epsilon import status_pb2

def get_values(prob):
    values = {}
    for var in prob.variables():
        values[var.id] = var.value
    return values

def assert_values_equal(a, b, tol=1e-2):
    assert a.keys() == b.keys()
    for var_id in a.keys():
        np.testing.assert_allclose(a[var_id], b[var_id], atol=tol)

def test_lasso():
    m = 5
    n = 10

    np.random.seed(0)
    A = np.random.randn(m,n)
    b = np.random.randn(m,1)
    x = cp.Variable(n,1)
    lam = 1

    f = 0.5*cp.sum_squares(A*x - b) + lam*cp.norm1(x)
    prob = cp.Problem(cp.Minimize(f))

    status = solve.solve(prob)
    assert status.state == status_pb2.ProblemStatus.OPTIMAL
    values0 = get_values(prob)

    prob.solve()
    values1 = get_values(prob)

    assert_values_equal(values0, values1)
