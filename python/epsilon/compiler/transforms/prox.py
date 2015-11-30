"""Transform a problem to prox-affine form.

TODO(mwytock): Clean up the interaction between expression matching and args
extraction.
"""

import logging

from epsilon import expression
from epsilon import tree_format
from epsilon.compiler.transforms import conic
from epsilon.compiler.transforms import linear
from epsilon.compiler.transforms.transform_util import *
from epsilon.expression_pb2 import Cone, Expression, ProxFunction, Problem, Size

class MatchResult(object):
    def __init__(self, match, prox_expr=None, constrs=[]):
        self.match = match
        self.prox_expr = prox_expr
        self.constrs = constrs

def convert_diagonal(expr):
    if not expr.dcp_props.affine:
        return epi_transform(expr, "affine")
    linear_expr = linear.transform_expr(expr)
    if linear_expr.affine_props.diagonal:
        return linear_expr, []
    return epi_transform(linear_expr, "diagonal")

def convert_scalar(expr):
    if not expr.dcp_props.affine:
        return epi_transform(expr, "affine")
    linear_expr = linear.transform_expr(expr)
    if linear_expr.affine_props.scalar:
        return linear_expr, []
    return epi_transform(linear_expr, "scalar")

def convert_affine(expr):
    if not expr.dcp_props.affine:
        return epi_transform(expr, "affine")
    return linear.transform_expr(expr), []

# Simple functions

def prox_constant(expr):
    if expr.dcp_props.constant:
        return MatchResult(
            True,
            expression.prox_function(
                ProxFunction(prox_function_type=ProxFunction.CONSTANT),
                linear.transform_expr(expr)))
    else:
        return MatchResult(False)


def prox_affine(expr):
    if expr.dcp_props.affine:
        return MatchResult(
            True,
            expression.prox_function(
                ProxFunction(prox_function_type=ProxFunction.AFFINE),
                linear.transform_expr(expr)))
    else:
        return MatchResult(False)

# Cones

def prox_zero(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    affine_arg, constrs = convert_affine(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.ZERO),
            affine_arg),
        constrs)

def prox_non_negative(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.NON_NEGATIVE):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.NON_NEGATIVE),
            diagonal_arg),
        constrs)

def prox_second_order_cone(expr):
    args = []
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.SECOND_ORDER):
        args = expr.arg
    else:
        f_expr, t_expr = get_epigraph(expr)
        if (f_expr and
            f_expr.expression_type == Expression.NORM_P and
            f_expr.p == 2):
            args = [t_expr, f_expr.arg[0]]
            # make second argument a row vector
            args[1] = expression.reshape(args[1], 1, dim(args[1]))
    if not args:
        return MatchResult(False)

    scalar_arg0, constrs0 = convert_scalar(args[0])
    scalar_arg1, constrs1 = convert_scalar(args[1])
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.SECOND_ORDER_CONE,
                arg_size=[
                    Size(dim=dims(args[0])),
                    Size(dim=dims(args[1]))]),
            scalar_arg0,
            scalar_arg1),
        constrs0 + constrs1)

def prox_semidefinite(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.SEMIDEFINITE):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.SEMIDEFINITE,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

# Elementwise

def prox_norm1(expr):
    pass

def prox_sum_deadzone(expr):
    pass

def prox_sum_exp(expr):
    pass

def prox_sum_square(expr):
    if (expr.expression_type == Expression.QUAD_OVER_LIN and
        expr.arg[1].expression_type == Expression.CONSTANT and
        expr.arg[1].constant.scalar == 1):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    affine_arg, constrs = convert_affine(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_SQUARE),
            affine_arg),
        constrs)

# Vector

def prox_max(expr):
    if expr.expression_type == Expression.MAX_ENTRIES:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.MAX),
            scalar_arg),
        constrs)

# Matrix

def prox_lambda_max(expr):
    if expr.expression_type == Expression.LAMBDA_MAX:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.LAMBDA_MAX,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

# Conic transform (catch-all default)

def transform_cone(expr):
    obj, constrs = conic.transform_expr(expr)
    return MatchResult(True, None, [obj] + constrs)

# Add rules in reverse priority order to apply high-level prox rules first
RULES = [
    # Matrix
    prox_lambda_max,

    # Vector
    prox_max,

    # Elementwise
    prox_sum_square,

    # Cone
    prox_second_order_cone,
    prox_non_negative,
    prox_semidefinite,
    prox_zero,

    # Simple
    prox_constant,
    prox_affine,

    # Lowest priority, transform to cone problem
    transform_cone
]

def transform_expr(expr):
    log_debug_expr("transform_expr", expr)
    for rule in RULES:
        result = rule(expr)

        if result.match:
            if result.prox_expr:
                yield result.prox_expr

            for constr in result.constrs:
                for f_expr in transform_expr(constr):
                    yield f_expr
            break
    else:
        raise TransformError("No rule matched")

def transform_problem(problem):
    f_exprs = list(transform_expr(problem.objective))
    for constr in problem.constraint:
        f_exprs += list(transform_expr(constr))
    return expression.Problem(objective=expression.add(*f_exprs))
