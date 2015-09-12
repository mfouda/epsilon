
from itertools import chain

from epsilon.expression_pb2 import Curvature, Expression

def is_elementwise(expr):
    if not expr.curvature.curvature_type == Curvature.AFFINE:
        return False

    if expr.expression_type == Expression.VARIABLE:
        return True

    if expr.expression_type in (
            Expression.MULTIPLY_ELEMENTWISE,
            Expression.ADD):
        return all(arg.curvature.elementwise or
                   arg.curvature.curvature_type == Curvature.CONSTANT
                   for arg in expr.arg)

    return False

def is_constant_multiple(expr):
    if expr.expression_type == Expression.VARIABLE:
        return True

    return False

def add_attributes(expr):
    """Add expression attributes helpful for translation."""
    for arg in expr.arg:
        add_attributes(arg)
    expr.curvature.elementwise = is_elementwise(expr)
    expr.curvature.constant_multiple = is_constant_multiple(expr)

def transform(problem):
    for expr in chain([problem.objective], problem.constraint):
        add_attributes(expr)
    return problem
