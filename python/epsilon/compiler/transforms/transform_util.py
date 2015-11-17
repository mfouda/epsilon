
import struct

from epsilon import error
from epsilon import expression
from epsilon import dcp
from epsilon.expression_pb2 import Expression, Curvature

class TransformError(error.ExpressionError):
    pass

def fp_expr(expr):
    return struct.pack("q", hash(expr.SerializeToString())).encode("hex")

def validate_args(expr, count):
    if len(expr.arg) != count:
        raise TransformError(
            "invalid args %d != %d" % (len(expr.args), count),
            expr)

def validate_size(expr, size):
    if expr.size.dim != size:
        raise TransformError(
            "invalid arg size %s != %s" % (expr.size.dim, size))

def only_arg(expr):
    validate_args(expr, 1)
    return expr.arg[0]

def dim(expr, index=None):
    if len(expr.size.dim) != 2:
        raise ExpressioneError("wrong number of dimensions", expr)
    if index is None:
        return expr.size.dim[0]*expr.size.dim[1]
    else:
        return expr.size.dim[index]

def epi(f_expr, t_expr):
    """An expression for an epigraph constraint.

    The constraint depends on the curvature of f:
      - f convex,  I(f(x) <= t)
      - f concave, I(f(x) >= t)
      - f affine,  I(f(x) == t)
    """
    f_curvature = dcp.get_curvature(f_expr)

    if f_curvature.curvature_type == Curvature.CONVEX:
        return expression.leq_constraint(f_expr, t_expr)
    elif f_curvature.curvature_type == Curvature.CONCAVE:
        return expression.leq_constraint(negate(f_expr), negate(t_expr))
    elif f_curvature.curvature_type == Curvature.AFFINE:
        return expression.eq_constraint(f_expr, t_expr);

    raise TransformError("Unknown curvature", f_expr)

def epi_var(expr, name, size=None):
    if size is None:
        size = expr.size.dim
    name += ":" + fp_expr(expr)
    return expression.variable(size[0], size[1], name)

def epi_transform(f_expr, name):
    t_expr = epi_var(f_expr, name)
    epi_f_expr = epi(f_expr, t_expr)
    return t_expr, epi_f_expr
