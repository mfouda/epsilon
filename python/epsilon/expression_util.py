
import struct

from epsilon.util import prod
from epsilon.expression_pb2 import Curvature, Expression

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
# TODO(mwytock): Put elsewhere
def dim(expr, index=0):
    pass

def only_arg(expr):
    pass
