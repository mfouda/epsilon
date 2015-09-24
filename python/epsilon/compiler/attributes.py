"""Compute attributes on expression trees.

These attributes are in addition to the standard DCP attributes which are
already computed by cvxpy.

TODO(mwytock): These routines should be smarter and able to distinguish between
x + y vs. x + x.
"""
from itertools import chain

from epsilon.expression import dimension
from epsilon.expression_pb2 import Curvature, Expression

def is_elementwise(expr):
    if expr.curvature.scalar_multiple:
        return True

    if (expr.expression_type ==
        Expression.MULTIPLY_ELEMENTWISE):
        return all(arg.curvature.elementwise for arg in expr.arg)
    return False

def is_scalar_expression(expr):
    if (expr.expression_type in (
            Expression.VARIABLE,
            Expression.CONSTANT,
            Expression.INDEX,
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


def compute_variable_curvature(expr):
    """Compute curvature attributes on per variable basis."""

    if expr.expression_type == Expression.VARIABLE:
        return {expr.variable.variable_id: Curvature(scalar_multiple=True)}

    retval = {}
    default = Curvature(scalar_multiple=is_scalar_expression(expr))

    for arg in expr.arg:
        for var_id, c in compute_variable_curvature(arg).iteritems():
            d = retval.get(var_id, default)
            retval[var_id] = Curvature(
                scalar_multiple=c.scalar_multiple and d.scalar_multiple)

    return retval
