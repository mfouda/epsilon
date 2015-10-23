
import numpy as np

from epsilon import data
from epsilon import expression
from epsilon import linear_map
from epsilon.compiler import canonicalize_linear
from epsilon.expression_str import expr_str
from epsilon.expression_testutil import assert_expr_equal

def test_index_vector_constant():
    c = data.store_constant(np.array([1,2,3]))
    assert_expr_equal(
        expression.linear_map(linear_map.index(slice(1, 2), 3), c),
        canonicalize_linear.transform_expr(
            expression.index(c, 1, 2)))
