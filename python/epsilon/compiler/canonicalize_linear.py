"""Implements the linear canonicalize transforms on the AST."""

from epsilon import expression
from epsilon import linear_map
from epsilon.compiler.canonicalize_util import *

# Transforms on the AST
def transform_variable(expr):
    return expr

def transform_constant(expr):
    return expr

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

def transform_multiply_generic(expr, f_const):
    if len(expr.arg) != 2:
        raise CanonicalizeError("wrong number of args", expr)

    if expr.arg[0].curvature.curvature_type == Curvature.CONSTANT:
        return expression.linear_map(
            linear_map.left_matrix_product(
                f_const(transform_expr(expr.arg[0])),
                dim(expr.arg[1], 1)),
            transform_expr(expr.arg(1)))
    elif expr.arg[1].curvature.curvature_type == Curvature.CONSTANT:
        return expression.linear_map(
            linear_map.right_matrix_product(
                f_const(transform_expr(expr.arg(1))),
                dim(expr.arg[0], 0)),
            transform_expr(expr.arg(0)))

    raise CanonicalizeError("multiplying two non constants", expr)

def constant_multiply(expr):
    leaf = only_leaf(expr)
    leaf.CopyFrom(
        expression.linear_map(linear_map.dense(leaf.constant)))
    return expr

def transform_multiply(expr):
    return transform_multiply_generic(expr, constant_multiply)

def constant_multiply_elementwise(expr):
    leaf = only_leaf(expr)
    leaf.CopyFrom(
        expression.linear_map(linear_map.diagonal(leaf.constant)))
    return expr

def transform_multiply_elementwise(expr):
    return transform_multiply_generic(expr, constant_multiply_elementwise)

def transform_negate(expr):
    return expression.linear_map(
        linear_map.negate(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_sum(expr):
    return expression.linear_map(
        linear_map.sum(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_linear_expr(expr):
    transform = locals()["transform_" + expr.expression_type().lower()]
    return transform(expr)

def transform_expr(expr):
    if expr.curvature.curvature_type == Curvature.AFFINE:
        return transform_linear_expr(expr)
    else:
        for arg in expr.arg:
            arg.CopyFrom(transform_expr(arg))
        return expr

def transform_problem(problem):
    validate.check_sum_of_prox(problem)
    f = [transform_expr(e) for e in problem.objective.arg]
    C = [transform_expr(e) for e in problem.constraint]
    return Problem(objective=add(f), constraint=C)
