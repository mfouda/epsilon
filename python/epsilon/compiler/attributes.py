"""Compute attributes on expression trees.

These attributes are in addition to the standard DCP attributes which are
already computed by cvxpy.

TODO(mwytock): These routines should be smarter and able to distinguish between
x + y vs. x + x.
"""
from itertools import chain

from epsilon.expression import dimension
from epsilon.expression_pb2 import Curvature, Expression

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

def compute_variable_curvature(expr):
    """Compute curvature attributes on per variable basis."""

    if expr.expression_type == Expression.VARIABLE:
        return {expr.variable.variable_id: Curvature(scalar_multiple=True)}

    retval = {}
    default = Curvature(scalar_multiple=(
        is_scalar_expression(expr) or
        expr.expression_type == Expression.ADD))

    for arg in expr.arg:
        for var_id, c in compute_variable_curvature(arg).iteritems():
            d = retval.get(var_id, default)
            retval[var_id] = Curvature(
                scalar_multiple=c.scalar_multiple and d.scalar_multiple)

    return retval
