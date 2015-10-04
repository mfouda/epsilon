"""Functional form of the expression operators."""

from epsilon.expression_pb2 import *
from epsilon.util import prod

# Internal helpers

def _add_binary(a, b):
    if prod(a.size.dim) == 1:
        size = b.size
    elif prod(b.size.dim) == 1:
        size = a.size
    elif a.size == b.size:
        size = a.size
    else:
        raise ValueError("adding incompatible sizes")

    curvature = Curvature()
    if a.curvature.curvature_type == Curvature.CONSTANT:
        curvature = b.curvature
    elif b.curvature.curvature_type == Curvature.CONSTANT:
        curvature = a.curvature
    elif a.curvature.curvature_type == Curvature.AFFINE:
        curvature.curvature_type = b.curvature.curvature_type
    elif b.curvature.curvature_type == Curvature.AFFINE:
        curvature.curvature_type = a.curvature.curvature_type
    elif a.curvature.curvature_type == b.curvature.curvature_type:
        curvature.curvature_type = a.curvature.curvature_type
    else:
        curvature.curvature_type = Curvature.UNKNOWN

    return Expression(curvature=curvature, size=size)

def _multiply_binary(a, b, elemwise=False):
    if prod(a.size.dim) == 1:
        size = b.size
    elif prod(b.size.dim) == 1:
        size = a.size
    elif not elemwise and a.size.dim[1] == b.size.dim[0]:
        size = Size(dim=[a.size.dim[0], b.size.dim[1]])
    elif elemwise and a.size == b.size:
        size = a.size
    else:
        raise ValueError("multiplying incompatible sizes")

    curvature = Curvature()
    if a.curvature.curvature_type == Curvature.CONSTANT:
        curvature.curvature_type = b.curvature.curvature_type
        curvature.scalar_multiple = prod(a.size.dim) == 1
        curvature.elementwise = curvature.scalar_multiple or elemwise
    elif b.curvature.curvature_type == Curvature.CONSTANT:
        curvature.curvature_type = a.curvature.curvature_type
        curvature.scalar_multiple = prod(b.size.dim) == 1
        curvature.elementwise = curvature.scalar_multiple or elemwise
    else:
        curvature.curvature_type = Curvature.UNKNOWN

    return Expression(curvature=curvature, size=size)

def _multiply(args, elemwise=False):
    if not args:
        raise ValueError("multiplying null args")

    a = args[0]
    for i in range(1, len(args)):
        a = _multiply_binary(a, args[i], elemwise)

    return Expression(
        expression_type=(Expression.MULTIPLY_ELEMENTWISE if elemwise else
                         Expression.MULTIPLY),
        arg=args,
        size=a.size,
        curvature=a.curvature)

# Expressions

def add(*args):
    if not args:
        raise ValueError("adding null args")

    a = args[0]
    for i in range(1, len(args)):
        a = _add_binary(a, args[i])

    return Expression(
        expression_type=Expression.ADD,
        arg=args,
        size=a.size,
        curvature=a.curvature)

def multiply(*args):
    return _multiply(args, elemwise=False)

def multiply_elemwise(*args):
    return _multiply(args, elemwise=True)

def hstack(*args):
    e = Expression(expression_type=Expression.HSTACK)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
            e.curvature.curvature_type = arg.curvature.curvature_type
            e.sign.sign_type = arg.sign.sign_type
        else:
            assert e.size.dim[0] == arg.size.dim[0]
            e.size.dim[1] += arg.size.dim[1]

            if arg.curvature.curvature_type != e.curvature.curvature_type:
                e.curvature.curvature_type = Curvature.UNKNOWN
            if arg.sign.sign_type != e.sign.sign_type:
                e.sign.sign_type = Sign.UNKNOWN

        e.arg.add().CopyFrom(arg)

    return e

def vstack(*args):
    e = Expression(expression_type=Expression.VSTACK)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
        else:
            assert e.size.dim[1] == arg.size.dim[1]
            e.size.dim[0] += arg.size.dim[0]

        e.arg.add().CopyFrom(arg)

    return e

def reshape(arg, m, n):
    assert m*n == prod(arg.size.dim)

    return Expression(
        expression_type=Expression.RESHAPE,
        arg=[arg],
        size=Size(dim=[m,n]),
        curvature=arg.curvature,
        sign=arg.sign)

def negate(arg):
    NEGATE_CURVATURE = {
        Curvature.UNKNOWN: Curvature.UNKNOWN,
        Curvature.AFFINE: Curvature.AFFINE,
        Curvature.CONVEX: Curvature.CONCAVE,
        Curvature.CONCAVE: Curvature.CONVEX,
        Curvature.CONSTANT: Curvature.CONSTANT,
    }
    return Expression(
        expression_type=Expression.NEGATE,
        arg=[arg],
        size=arg.size,
        curvature=Curvature(
            curvature_type=NEGATE_CURVATURE[arg.curvature.curvature_type],
            elementwise=arg.curvature.elementwise,
            scalar_multiple=arg.curvature.scalar_multiple))

def variable(m, n, variable_id):
    return Expression(
        expression_type=Expression.VARIABLE,
        size=Size(dim=[m, n]),
        variable=Variable(variable_id=variable_id),
        curvature=Curvature(
            curvature_type=Curvature.AFFINE,
            elementwise=True,
            scalar_multiple=True))

def constant(m, n, scalar=None, data_location=None):
    if scalar is not None:
        constant = Constant(scalar=scalar)
    elif data_location is not None:
        constant = Constant(data_location=data_location)
    else:
        raise ValueError("need either scalar or data_location")

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

def abs_val(x):
    return Expression(
        expression_type=Expression.ABS,
        size=x.size,
        arg=[x])

def sum_entries(x):
    return Expression(
        expression_type=Expression.SUM,
        size=Size(dim=[1, 1]),
        arg=[x])

def transpose(x):
    m, n = x.size.dim
    return Expression(
        expression_type=Expression.TRANSPOSE,
        size=Size(dim=[n, m]),
        curvature=x.curvature,
        arg=[x])

def equality_constraint(a, b):
    return indicator(Cone.ZERO, add(a, negate(b)))

def leq_constraint(a, b):
    return indicator(Cone.NON_NEGATIVE, add(b, negate(a)))

def non_negative(x):
    return indicator(Cone.NON_NEGATIVE, x)
