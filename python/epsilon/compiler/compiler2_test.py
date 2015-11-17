
from nose.tools import assert_items_equal, assert_equal

from epsilon import cvxpy_expr
from epsilon.compiler import compiler2 as compiler
from epsilon.compiler import validate
from epsilon.problems import basis_pursuit
from epsilon.problems import least_abs_dev
from epsilon.problems import tv_1d
from epsilon.problems import tv_denoise
from epsilon.expression_pb2 import Expression, ProxFunction

Prox = ProxFunction

# temporary debugging
import logging
logging.basicConfig(level=logging.DEBUG)


def prox_ops(expr):
    retval = []
    for arg in expr.arg:
        retval += prox_ops(arg)
    if expr.expression_type == Expression.PROX_FUNCTION:
        retval.append(expr.prox_function.prox_function_type)
    return retval

def test_basis_pursuit():
    problem = compiler.compile_problem(cvxpy_expr.convert_problem(
        basis_pursuit.create(m=10, n=30)))
    assert_items_equal(
        prox_ops(problem.objective),
        [Prox.ZERO, Prox.AFFINE] + 2*[Prox.NON_NEGATIVE])
    assert_equal(0, len(problem.constraint))

# def test_least_abs_deviations():
#     problem = compiler.compile_problem(cvxpy_expr.convert_problem(
#         least_abs_dev.create(m=10, n=5)))
#     assert_items_equal(prox_ops(problem), ["NormL1Prox", "ZeroProx"])
#     assert_equal(1, len(problem.constraint))

# def test_tv_denoise():
#     problem = compiler.compile_problem(cvxpy_expr.convert_problem(
#         tv_denoise.create(n=10, lam=1)))
#     assert_items_equal(
#         prox_ops(problem), 3*["LeastSquaresProx"] + ["NormL1L2Prox"] + ["LinearEqualityProx"])
#     assert_equal(4, len(problem.constraint))

# def test_tv_1d():
#     problem = compiler.compile_problem(cvxpy_expr.convert_problem(
#         tv_1d.create(n=10)))
#     assert_items_equal(
#         prox_ops(problem), ["LeastSquaresProx", "FusedLassoProx"])
#     assert_equal(1, len(problem.constraint))
