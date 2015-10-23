"""Implements the linear canonicalize transforms on the AST."""

from epsilon import expression
from epsilon import linear_map
from epsilon.compiler import validate
from epsilon.expression_pb2 import Problem, Constant
from epsilon.expression_util import *

# Transforms on the AST
def transform_variable(expr):
    if dim(expr,1) == 1:
        return expr
    return expression.reshape(expr, dim(expr), 1)

def transform_constant(expr):
    if dim(expr,1) == 1:
        return expr
    return expression.reshape(expr, dim(expr), 1)

def transform_add(expr):
    return expression.add(*(transform_expr(e) for e in expr.arg))

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

def transform_multiply_generic(expr, const_transform):
    if len(expr.arg) != 2:
        raise CanonicalizeError("wrong number of args", expr)

    m = dim(expr, 0)
    n = dim(expr, 1)
    k = dim(expr.arg[0], 1)

    if expr.arg[0].curvature.curvature_type == Curvature.CONSTANT:
        return expression.linear_map(
            linear_map.left_matrix_product(const_transform(expr.arg[0], k), n),
            transform_expr(expr.arg[1]))

    if expr.arg[1].curvature.curvature_type == Curvature.CONSTANT:
        return expression.linear_map(
            linear_map.right_matrix_product(const_transform(expr.arg[1], k), m),
            tranform_expr(expr.arg[0]))

    raise CanonicalizeError("multiplying non constants", expr)

def multiply_const_transform(expr, n):
    # TODO(mwytock): Handle this case
    if expr.expression_type != Expression.CONSTANT:
        raise CanonicalizeError("multiply constant is not leaf", expr)

    if expr.constant.constant_type == Constant.SCALAR:
        return linear_map.scalar(expr.constant.scalar, n)
    if expr.constant.constant_type == Constant.DENSE_MATRIX:
        return linear_map.dense_matrix(expr.constant)
    if expr.constant.constant_type == Constant.SPARSE_MATRIX:
        return linear_map.sparse_matrix(expr.constant)

    raise CanonicalizeError("unknown constant type", expr)

def transform_multiply(expr):
    return transform_multiply_generic(expr, multiply_const_transform)

def transform_negate(expr):
    return expression.linear_map(
        linear_map.negate(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_sum(expr):
    return expression.linear_map(
        linear_map.sum(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_linear_expr(expr):
    f_name = "transform_" + Expression.Type.Name(expr.expression_type).lower()
    return globals()[f_name](expr)

def transform_expr(expr):
    if expr.curvature.curvature_type in (Curvature.AFFINE, Curvature.CONSTANT):
        return transform_linear_expr(expr)
    else:
        for arg in expr.arg:
            arg.CopyFrom(transform_expr(arg))
        return expr

def transform_problem(problem):
    validate.check_sum_of_prox(problem)
    f = [transform_expr(e) for e in problem.objective.arg]
    C = [transform_expr(e) for e in problem.constraint]
    return Problem(objective=expression.add(*f), constraint=C)
