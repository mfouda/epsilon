"""Compute attributes on expression trees.

These attributes are in addition to the standard DCP attributes which are
already computed by cvxpy.
"""
from itertools import chain

from epsilon.expression import dimension
from epsilon.expression_pb2 import Curvature, Expression

def is_elementwise(expr):
    if expr.curvature.scalar_multiple:
        return True

    if expr.expression_type in (
            Expression.MULTIPLY_ELEMENTWISE,
            Expression.ADD):
        return all(arg.curvature.elementwise for arg in expr.arg)
    return False

def is_scalar_expression(expr):
    if (expr.expression_type in (
            Expression.VARIABLE,
            Expression.CONSTANT,
            Expression.INDEX,
            Expression.ADD,
            Expression.NEGATE)):
        return True

    if (expr.expression_type == Expression.MULTIPLY and
        dimension(expr.arg[0]) == 1):
        return True

    return False

def is_scalar_multiple(expr):
    return (is_scalar_expression(expr) and
            all(arg.curvature.scalar_multiple for arg in expr.arg))

def add_attributes(expr):
    """Add expression attributes helpful for translation."""
    for arg in expr.arg:
        add_attributes(arg)
    expr.curvature.scalar_multiple = is_scalar_multiple(expr)
    expr.curvature.elementwise = is_elementwise(expr)

def transform(problem):
    for expr in chain([problem.objective], problem.constraint):
        add_attributes(expr)
    return problem
