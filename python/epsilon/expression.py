"""Functional form of the expression operators."""

import numpy as np

from epsilon import constant as _constant
from epsilon.error import ExpressionError
from epsilon.expression_pb2 import *
from epsilon.expression_util import *

# Shorthand convenience
SIGNED = Monotonicity(monotonicity_type=Monotonicity.SIGNED)

AFFINE = Curvature(curvature_type=Curvature.AFFINE)
CONSTANT = Curvature(curvature_type=Curvature.CONSTANT)

def is_scalar(a):
    return a[0]*a[1] == 1

def elementwise_dims(a, b):
    if a == b:
        return a
    if is_scalar(b):
        return a
    if is_scalar(a):
        return b
    raise ValueError("Incompatible elemwise binary op sizes")

def matrix_multiply_dims(a, b):
    if a[1] == b[0]:
        return (a[0], b[1])
    if is_scalar(a):
        return b
    if is_scalar(b):
        return a
    raise ValueError("Incompatible matrix multiply sizes")

def _multiply(args, elemwise=False):
    if not args:
        raise ValueError("multiplying null args")

    op_dims = elementwise_dims if elemwise else matrix_multiply_dims
    return Expression(
        expression_type=(Expression.MULTIPLY_ELEMENTWISE if elemwise else
                         Expression.MULTIPLY),
        arg=args,
        size=Size(dim=reduce(lambda a, b: op_dims(a, b),
                             (dims(a) for a in args))),
        curvature=AFFINE)

# Expressions
def add(*args):
    if not args:
        raise ValueError("adding null args")

    return Expression(
        expression_type=Expression.ADD,
        arg=args,
        size=Size(
            dim=reduce(lambda a, b: elementwise_dims(a, b),
                       (dims(a) for a in args))),
        curvature=AFFINE)

def multiply(*args):
    return _multiply(args, elemwise=False)

def multiply_elemwise(*args):
    return _multiply(args, elemwise=True)

def hstack(*args):
    e = Expression(
        expression_type=Expression.HSTACK,
        curvature=AFFINE)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
        else:
            if dim(e, 0) != dim(arg, 0):
                raise ExpressionError("Incompatible sizes", e, arg)
            e.size.dim[1] += arg.size.dim[1]

        e.arg.add().CopyFrom(arg)

    return e

def vstack(*args):
    e = Expression(
        expression_type=Expression.VSTACK,
        curvature=AFFINE)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
        else:
            if dim(e, 1) != dim(arg, 1):
                raise ExpressionError("Incompatible sizes", e, arg)
            e.size.dim[0] += arg.size.dim[0]

        e.arg.add().CopyFrom(arg)

    return e

def reshape(arg, m, n):
    if dim(arg, 0) == m and dim(arg, 1) == n:
        return arg

    if m*n != dim(arg):
        raise ExpressionError("cant reshape to %d x %d" % (m, n), arg)

    # If we have two reshapes that "undo" each other, cancel them out
    if (arg.expression_type == Expression.RESHAPE and
        dim(arg.arg[0], 0) == m and
        dim(arg.arg[0], 1) == n):
        return arg.arg[0]

    return Expression(
        expression_type=Expression.RESHAPE,
        arg=[arg],
        size=Size(dim=[m,n]),
        curvature=arg.curvature,
        sign=arg.sign)

def negate(x):
    # Automatically reduce negate(negate(x)) to x
    if x.expression_type == Expression.NEGATE:
        return only_arg(x)

    return Expression(
        expression_type=Expression.NEGATE,
        arg=[x],
        size=x.size,
        curvature=AFFINE)

def variable(m, n, variable_id):
    return Expression(
        expression_type=Expression.VARIABLE,
        size=Size(dim=[m, n]),
        variable=Variable(variable_id=variable_id),
        curvature=Curvature(
            curvature_type=Curvature.AFFINE,
            elementwise=True,
            scalar_multiple=True))

def scalar_constant(scalar, size=None):
    if size is None:
        size = (1, 1)

    return Expression(
        expression_type=Expression.CONSTANT,
        size=Size(dim=size),
        constant=Constant(
            constant_type=Constant.SCALAR,
            scalar=scalar),
        curvature=CONSTANT)

def ones(*dims):
    return Expression(
        expression_type=Expression.CONSTANT,
        size=Size(dim=dims),
        constant=_constant.store(np.ones(dims)),
        curvature=CONSTANT)

def constant(m, n, scalar=None, constant=None):
    if scalar is not None:
        constant = Constant(
            constant_type=Constant.SCALAR,
            scalar=scalar)
    elif constant is None:
        raise ValueError("need either scalar or constant")

    return Expression(
        expression_type=Expression.CONSTANT,
        size=Size(dim=[m, n]),
        constant=constant,
        curvature=Curvature(curvature_type=Curvature.CONSTANT))

def indicator(cone_type, *args):
    return Expression(
        expression_type=Expression.INDICATOR,
        size=Size(dim=[1, 1]),
        cone=Cone(cone_type=cone_type),
        arg=args)

def norm_pq(x, p, q):
    return Expression(
        expression_type=Expression.NORM_PQ,
        size=Size(dim=[1, 1]),
        arg=[x], p=p, q=q)

def norm_p(x, p):
    return Expression(
        expression_type=Expression.NORM_P,
        size=Size(dim=[1, 1]),
        arg=[x], p=p)

def power(x, p):
    return Expression(
        expression_type=Expression.POWER,
        size=x.size,
        arg=[x], p=p)

def sum_largest(x, k):
    return Expression(
        expression_type=Expression.SUM_LARGEST,
        size=Size(dim=[1,1]),
        arg=[x], k=k)

def abs_val(x):
    return Expression(
        expression_type=Expression.ABS,
        arg_monotonicity=[SIGNED],
        size=x.size,
        arg=[x])

def sum_entries(x):
    return Expression(
        expression_type=Expression.SUM,
        size=Size(dim=[1, 1]),
        curvature=AFFINE,
        arg=[x])

def transpose(x):
    m, n = x.size.dim
    return Expression(
        expression_type=Expression.TRANSPOSE,
        size=Size(dim=[n, m]),
        curvature=AFFINE,
        arg=[x])

def trace(X):
    return Expression(
        expression_type=Expression.TRACE,
        size=Size(dim=[1, 1]),
        curvature=AFFINE,
        arg=[X])

def diag_vec(x):
    if dim(x, 1) != 1:
        raise ExpressionError("diag_vec on non vector")

    n = dim(x, 0)
    return Expression(
        expression_type=Expression.DIAG_VEC,
        size=Size(dim=[n, n]),
        curvature=AFFINE,
        arg=[x])

def index(x, start_i, stop_i, start_j=None, stop_j=None):
    if start_j is None and stop_j is None:
        start_j = 0
        stop_j = x.size.dim[1]

    if (dim(x, 0) == stop_i - start_i and
        dim(x, 1) == stop_j - start_j):
        return x

    return Expression(
        expression_type=Expression.INDEX,
        size=Size(dim=[stop_i-start_i, stop_j-start_j]),
        curvature=AFFINE,
        key=[Slice(start=start_i, stop=stop_i, step=1),
             Slice(start=start_j, stop=stop_j, step=1)],
        arg=[x])

def scaled_zone(x, alpha, beta, C, M):
    return Expression(
        expression_type=Expression.SCALED_ZONE,
        size=Size(dim=[1, 1]),
        scaled_zone_params=Expression.ScaledZoneParams(
            alpha=alpha,
            beta=beta,
            c=C,
            m=M),
        arg=[x])

def zero(x):
    return Expression(
        expression_type=Expression.ZERO,
        size=Size(dim=[1, 1]),
        arg=[x])

def linear_map(A, x):
    if dim(x, 1) != 1:
        raise ExpressionError("applying linear map to non vector", x)
    if A.n != dim(x):
        raise ExpressionError("linear map has wrong size: %s" % A, x)

    return Expression(
        expression_type=Expression.LINEAR_MAP,
        size=Size(dim=[A.m, 1]),
        curvature=AFFINE,
        linear_map=A,
        arg=[x])

def eq_constraint(x, y):
    if dims(x) != dims(y) and dim(x) != 1 and dim(y) != 1:
        raise ExpressionError("incompatible sizes", x, y)
    return indicator(Cone.ZERO, add(x, negate(y)))

def leq_constraint(a, b):
    return indicator(Cone.NON_NEGATIVE, add(b, negate(a)))

def soc_constraint(t, x):
    if dim(t) != 1:
        raise ExpressionError("Second order cone, dim(t) != 1", t)

    # make x a row vector to be compatible with elemwise version
    return indicator(Cone.SECOND_ORDER, t, reshape(x, 1, dim(x)))

def soc_elemwise_constraint(t, *args):
    t = reshape(t, dim(t), 1)
    X = hstack(*(reshape(arg, dim(arg), 1) for arg in args))
    if dim(t) != dim(X, 0):
        raise ExpressionError("Second order cone, incompatible sizes", t, X)
    return indicator(Cone.SECOND_ORDER, t, X)

def psd_constraint(X, Y):
    return indicator(Cone.SEMIDEFINITE, add(X, negate(Y)))

def semidefinite(X):
    return indicator(Cone.SEMIDEFINITE, X)

def non_negative(x):
    return indicator(Cone.NON_NEGATIVE, x)

def prox_function(f, *args):
    return Expression(
        expression_type=Expression.PROX_FUNCTION,
        size=Size(dim=[1, 1]),
        prox_function=f,
        arg=args)
