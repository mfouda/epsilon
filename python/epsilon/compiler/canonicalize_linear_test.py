
import logging

import numpy as np

from epsilon import data
from epsilon import expression
from epsilon import linear_map
from epsilon import tree_format
from epsilon.compiler import canonicalize_linear
from epsilon.expression_testutil import assert_expr_equal

c = data.store_constant(np.array([1,2,3]))
C = data.store_constant(np.array([[1,2,3],[4,5,6]]))

x = expression.variable(3, 1, "x")
X = expression.variable(3, 4, "X")

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
     expression.multiply(C, x)),
    ("multiply_matrix",
     expression.multiply(C, X),
     expression.multiply(C, X)),
]

def _test(name, expr, expected):
    logging.debug("Input:\n%s", tree_format.format_expr(expr))
    assert_expr_equal(expected, canonicalize_linear.transform_expr(expr))

def test():
    for name, expr, expected in TESTS:
        yield _test, name, expr, expected
