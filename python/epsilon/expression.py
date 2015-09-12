"""Functional form of the expression operators."""

from epsilon.expression_pb2 import *

# Accessors

def scalar_constant(expr):
    if len(expr.arg) == 0:
        assert expr.expression_type == Expression.CONSTANT
        assert dimension(expr) == 1
        return expr.constant.scalar

    assert False, "not implemented"

def dimension(expr):
    return expr.size.dim[0]*expr.size.dim[1]

def expr_vars(expr):
    retval = {}
    if expr.expression_type == Expression.VARIABLE:
        retval[expr.variable.variable_id] = expr
    else:
        for arg in expr.arg:
            retval.update(expr_vars(arg))
    return retval

# Constructors

def add(*args):
    assert len(args) > 0
    for i in range(len(args),1):
        assert args[0].size == args[i].size

    return Expression(
        expression_type=Expression.ADD,
        arg=args,
        size=args[0].size)

def multiply(*args):
    assert len(args) == 2
    assert args[0].size.dim[1] == args[1].size.dim[0]

    return Expression(
        expression_type=Expression.MULTIPLY,
        arg=args,
        size=Size(dim=[args[0].size.dim[0], args[1].size.dim[1]]))

def hstack(*args):
    e = Expression(expression_type=Expression.HSTACK)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
        else:
            assert e.size.dim[0] == arg.size.dim[0]
            e.size.dim[1] += arg.size.dim[1]

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
    assert m*n == dimension(arg)

    return Expression(
        expression_type=Expression.RESHAPE,
        arg=[arg],
        size=Size(dim=[m,n]))

def negate(arg):
    return Expression(
        expression_type=Expression.NEGATE,
        arg=[arg],
        size=arg.size)

def variable(m, n, variable_id):
    return Expression(
        expression_type=Expression.VARIABLE,
        size=Size(dim=[m, n]),
        variable=Variable(variable_id=variable_id))

def constant(m, n, scalar=0):
    return Expression(
        expression_type=Expression.CONSTANT,
        size=Size(dim=[m, n]),
        constant=Constant(scalar=scalar))
