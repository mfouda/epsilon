
import numpy as np

from epsilon import data
from epsilon import expression
from epsilon import linear_map
from epsilon.compiler import canonicalize_linear

def assert_expressions_equal(a, b):
    pass

def test_index_vector_constant():
    c = data.store_constant(np.array([1,2,3]))
    assert_expressions_equal(
        expression.linear_map(
            linear_map.sparse_matrix(
                linear_map.index(slice(1, 2), 3)), c),
        canonicalize_linear.transform_expr(
            expression.index(1, 2), c))
