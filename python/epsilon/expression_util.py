
import struct

from epsilon.error import ExpressionError
from epsilon.expression_pb2 import Curvature, Expression
from epsilon.util import prod

def fp_expr(expr):
    return struct.pack("q", hash(expr.SerializeToString())).encode("hex")

def is_scalar_expression(expr):
    if (expr.expression_type in (
            Expression.VARIABLE,
            Expression.CONSTANT,
            Expression.NEGATE)):
        return True

    if (expr.expression_type == Expression.MULTIPLY and
        prod(expr.arg[0].size.dim) == 1):
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

def expr_vars(expr):
    if expr.expression_type == Expression.VARIABLE:
        return {expr.variable.variable_id}

    retval = set()
    for arg in expr.arg:
        retval |= expr_vars(arg)
    return retval

# Helper functions
def only_arg(expr):
    if len(expr.arg) != 1:
        raise ExpressionError("wrong number of args", expr)
    return expr.arg[0]

def dim(expr, index=None):
    if len(expr.size.dim) != 2:
        raise ExpressioneError("wrong number of dimensions", expr)
    if index is None:
        return expr.size.dim[0]*expr.size.dim[1]
    else:
        return expr.size.dim[index]
