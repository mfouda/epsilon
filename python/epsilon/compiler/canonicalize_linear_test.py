
import logging

import numpy as np

from epsilon import constant
from epsilon import expression
from epsilon import linear_map
from epsilon import tree_format
from epsilon.compiler import canonicalize_linear
from epsilon.expression_testutil import assert_expr_equal

c = expression.constant(
    3, 1, constant=constant.store(np.array([1,2,3])))
C = expression.constant(
    2, 3, constant=constant.store(np.array([[1,2,3],[4,5,6]])))

x = expression.variable(3, 1, "x")
y = expression.variable(3, 1, "y")
X = expression.variable(3, 4, "X")
Y = expression.variable(3, 2, "Y")

TESTS = [
    ("index_constant",
     expression.index(c, 1, 2),
     expression.linear_map(linear_map.index(slice(1, 2), 3), c)),
    ("index_matrix_constant",
     expression.index(C, 0, 1, 0, 2),
     expression.linear_map(
         linear_map.kronecker_product(
             linear_map.index(slice(0, 2), 3),
             linear_map.index(slice(0, 1), 2)),
         expression.reshape(C, 6, 1))),
    ("multiply_vector",
     expression.multiply(C, x),
     expression.linear_map(
         linear_map.dense_matrix(C.constant),
         x)),
    ("multiply_matrix",
     expression.multiply(C, X),
     expression.linear_map(
         linear_map.kronecker_product(
             linear_map.identity(4),
             linear_map.dense_matrix(C.constant)),
         expression.reshape(X, 12, 1))),
    ("hstack_vector",
     expression.hstack(x, y),
     expression.add(
         expression.linear_map(
             linear_map.right_matrix_product(
                 linear_map.index(slice(0, 1), 2), 3),
             x),
         expression.linear_map(
             linear_map.right_matrix_product(
                 linear_map.index(slice(1, 2), 2), 3),
             y))),
    ("hstack_matrix",
     expression.hstack(X, Y),
     expression.add(
         expression.linear_map(
             linear_map.right_matrix_product(
                 linear_map.index(slice(0, 4), 6), 3),
             expression.reshape(X, 12, 1)),
         expression.linear_map(
             linear_map.right_matrix_product(
                 linear_map.index(slice(4, 6), 6), 3),
             expression.reshape(Y, 6, 1)))),
]

def _test(name, expr, expected):
    logging.debug("Input:\n%s", tree_format.format_expr(expr))
    assert_expr_equal(expected, canonicalize_linear.transform_expr(expr))

def test():
    for name, expr, expected in TESTS:
        yield _test, name, expr, expected
