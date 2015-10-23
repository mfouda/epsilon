
from epsilon.compiler import canonicalize_linear
from epsilon import expression

def assert_expressions_equal(a, b):
    pass

def constant_vector(a):
    pass

def constant_matrix(A):
    pass

def test_index_vector_constant():
    assert_expressions_equal(
        expression.linear_map(),
        canonicalize_linear.transform_expr(
            expression.index(1, 2),
            constant_vector([1,2,3])))
