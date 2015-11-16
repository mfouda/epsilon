
import struct

from epsilon import error

class TransformError(error.ExpressionError):
    pass

def fp_expr(expr):
    return struct.pack("q", hash(expr.SerializeToString())).encode("hex")

def validate_args(expr, count):
    if len(expr.args) != count:
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

def epi_var(expr, name, size=None):
    if size is None:
        size = expr.size.dim
    name += ":" + fp_expr(expr)
    return expression.variable(size[0], size[1], name)
