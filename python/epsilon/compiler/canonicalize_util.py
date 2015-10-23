
from epsilon import error

class CanonicalizeError(error.ExpressionError):
    pass

# Helper functions
# TODO(mwytock): Move to cannicalize_util.py
def only_arg(expr):
    if len(expr.arg) != 1:
        raise CanonicalizeError("wrong number of args", expr)
    return expr.arg[0]

def dim(expr, index=None):
    if len(expr.size.dim) != 2:
        raise CanonicalizeError("wrong number of dimensions", expr)
    if index is None:
        return expr.size.dim[0]*expr.size.dim[1]
    else:
        return expr.size.dim[index]
