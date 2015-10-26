"""Implements the linear canonicalize transforms on the AST."""

from epsilon import error
from epsilon import expression
from epsilon import linear_map
from epsilon.compiler import validate
from epsilon.expression_pb2 import Problem, Constant
from epsilon.expression_util import *

class CanonicalizeError(error.ExpressionError):
    pass

# Transforms on the AST
def transform_variable(expr):
    if dim(expr,1) == 1:
        return expr
    return expression.reshape(expr, dim(expr), 1)

def transform_constant(expr):
    if dim(expr,1) == 1:
        return expr
    return expression.reshape(expr, dim(expr), 1)

def maybe_promote(dim_sum, expr):
    if dim(expr) == 1 and dim_sum != 1:
        return expression.linear_map(linear_map.promote(dim_sum), expr)
    else:
        return expr

def transform_add(expr):
    dim_sum = dim(expr)
    return expression.add(
        *[maybe_promote(dim_sum, transform_expr(e)) for e in expr.arg])

def transform_transpose(expr):
    return expression.linear_map(
        linear_map.transpose(dim(expr,0), dim(expr,1)),
        transform_expr(only_arg(expr)))

def transform_index(expr):
    return expression.linear_map(
        linear_map.kronecker_product(
            linear_map.index(expr.key[1], dim(only_arg(expr),1)),
            linear_map.index(expr.key[0], dim(only_arg(expr),0))),
        transform_expr(only_arg(expr)))

def multiply_constant(expr, n):
    # TODO(mwytock): Handle this case
    if expr.expression_type == Expression.CONSTANT:
        if expr.constant.constant_type == Constant.SCALAR:
            return linear_map.scalar(expr.constant.scalar, n)
        if expr.constant.constant_type == Constant.DENSE_MATRIX:
            return linear_map.dense_matrix(expr.constant)
        if expr.constant.constant_type == Constant.SPARSE_MATRIX:
            return linear_map.sparse_matrix(expr.constant)
    elif expr.expression_type == Expression.TRANSPOSE:
        return linear_map.transpose(multiply_constant(only_arg(expr), n))

    raise CanonicalizeError("unknown constant type", expr)

def transform_multiply(expr):
    if len(expr.arg) != 2:
        raise CanonicalizeError("wrong number of args", expr)

    m = dim(expr, 0)
    n = dim(expr, 1)
    if expr.arg[0].curvature.curvature_type == Curvature.CONSTANT:
        return expression.linear_map(
            linear_map.left_matrix_product(
                multiply_constant(expr.arg[0], m), n),
            transform_expr(expr.arg[1]))

    if expr.arg[1].curvature.curvature_type == Curvature.CONSTANT:
        return expression.linear_map(
            linear_map.right_matrix_product(
                multiply_constant(expr.arg[1], n), m),
            transform_expr(expr.arg[0]))

    raise CanonicalizeError("multiplying non constants", expr)

def multiply_elementwise_constant(expr):
    # TODO(mwytock): Handle this case
    if expr.expression_type != Expression.CONSTANT:
        raise CanonicalizeError("multiply constant is not leaf", expr)

    if expr.constant.constant_type == Constant.DENSE_MATRIX:
        return linear_map.diagonal_matrix(expr.constant)

    raise CanonicalizeError("unknown constant type", expr)

def transform_multiply_elementwise(expr):
    if len(expr.arg) != 2:
        raise CanonicalizeError("wrong number of args", expr)

    if expr.arg[0].curvature.curvature_type == Curvature.CONSTANT:
        c_expr = expr.arg[0]
        x_expr = expr.arg[1]
    elif expr.arg[1].curvature.curvature_type == Curvature.CONSTANT:
        c_expr = expr.arg[1]
        x_expr = expr.arg[0]
    else:
        raise CanonicalizeError("multiply non constants", expr)

    return expression.linear_map(
        multiply_elementwise_constant(c_expr),
        transform_expr(x_expr))

def transform_negate(expr):
    return expression.linear_map(
        linear_map.negate(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_sum(expr):
    return expression.linear_map(
        linear_map.sum(dim(only_arg(expr))),
        transform_expr(only_arg(expr)))

def transform_hstack(expr):
    m = dim(expr, 0)
    n = dim(expr, 1)
    offset = 0
    add_args = []
    for arg in expr.arg:
        ni = dim(arg, 1)
        add_args.append(
            expression.linear_map(
                linear_map.right_matrix_product(
                    linear_map.index(slice(offset, offset+ni), n), m),
                transform_expr(arg)))
        offset += ni
    return expression.add(*add_args)

def transform_reshape(expr):
    return expression.reshape(
        transform_expr(only_arg(expr)),
        dim(expr, 0),
        dim(expr, 1))

def transform_linear_expr(expr):
    f_name = "transform_" + Expression.Type.Name(expr.expression_type).lower()
    return globals()[f_name](expr)

def transform_expr(expr):
    if expr.curvature.curvature_type in (Curvature.AFFINE, Curvature.CONSTANT):
        out_expr = transform_linear_expr(expr)
        if expr.HasField("proximal_operator"):
            out_expr.proximal_operator.CopyFrom(expr.proximal_operator)
        return out_expr
    else:
        for arg in expr.arg:
            arg.CopyFrom(transform_expr(arg))
        return expr

def transform_problem(problem):
    validate.check_sum_of_prox(problem)
    f = [transform_expr(e) for e in problem.objective.arg]
    C = [transform_expr(e) for e in problem.constraint]
    return Problem(objective=expression.add(*f), constraint=C)
