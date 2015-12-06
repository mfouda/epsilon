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

# Elementwise

def prox_norm_1(expr):
    if (expr.expression_type == Expression.NORM_P and
        expr.p == 1):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.NORM_1),
            diagonal_arg),
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

def prox_sum_deadzone(expr):
    hinge_arg = get_hinge_arg(expr)
    arg = None
    if (hinge_arg and
        hinge_arg.expression_type == Expression.ADD and
        len(hinge_arg.arg) == 2 and
        hinge_arg.arg[0].expression_type == Expression.ABS):
        m = get_scalar_constant(hinge_arg.arg[1])
        if m <= 0:
            arg = hinge_arg.arg[0].arg[0]
    if not arg:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.SUM_DEADZONE,
                scaled_zone_params=ProxFunction.ScaledZoneParams(m=-m)),
            diagonal_arg),
        constrs)

def prox_sum_hinge(expr):
    arg = get_hinge_arg(expr)
    if not arg:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_HINGE),
            diagonal_arg),
        constrs)

def prox_sum_quantile(expr):
    arg = None
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        len(expr.arg[0].arg) == 2):

        alpha, x = get_quantile_arg(expr.arg[0].arg[0])
        beta, y  = get_quantile_arg(expr.arg[0].arg[1])
        if (x is not None and y is not None and x == y and
            abs(alpha) <= 1 and abs(beta) <= 1):
            if alpha <= 0 and beta > 0:
                arg = x
                alpha, beta = beta, abs(alpha)
            elif beta <= 0 and alpha > 0:
                arg = x
                beta = abs(beta)
    if not arg:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.SUM_QUANTILE,
                scaled_zone_params=ProxFunction.ScaledZoneParams(
                    alpha=alpha,
                    beta=beta)),
            diagonal_arg),
        constrs)

def prox_sum_exp(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.EXP):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_EXP),
            diagonal_arg),
        constrs)

def prox_sum_inv_pos(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.POWER and
        expr.arg[0].p == -1):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_INV_POS),
            diagonal_arg),
        constrs)

def prox_sum_logistic(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.LOGISTIC):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_LOGISTIC),
            diagonal_arg),
        constrs)

def prox_sum_neg_entr(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.NEGATE and
        expr.arg[0].arg[0].expression_type == Expression.ENTR):
        arg = expr.arg[0].arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_NEG_ENTR),
            diagonal_arg),
        constrs)

def prox_sum_neg_log(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.NEGATE and
        expr.arg[0].arg[0].expression_type == Expression.LOG):
        arg = expr.arg[0].arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.SUM_NEG_LOG),
            diagonal_arg),
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

def prox_norm_2(expr):
    if expr.expression_type == Expression.NORM_P and expr.p == 2:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.NORM_2),
            scalar_arg),
        constrs)

def prox_sum_largest(expr):
    if expr.expression_type == Expression.SUM_LARGEST:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.SUM_LARGEST,
                sum_largest_params=ProxFunction.SumLargestParams(k=expr.k)),
            scalar_arg),
        constrs)

def prox_total_variation_1d(expr):
    arg = get_total_variation_arg(expr)
    if arg is None:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.TOTAL_VARIATION_1D),
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

def prox_neg_log_det(expr):
    if (expr.expression_type == Expression.NEGATE and
        expr.arg[0].expression_type == Expression.LOG_DET):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.NEG_LOG_DET,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

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

def prox_norm_nuclear(expr):
    if expr.expression_type == Expression.NORM_NUC:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            ProxFunction(
                prox_function_type=ProxFunction.NORM_NUCLEAR,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

# Any affine function

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

# Epigraph transform

def epigraph(expr):
    f_expr, t_expr = get_epigraph(expr)
    if f_expr:
        for rule in BASE_RULES:
            result = rule(f_expr)

            if result.match:
                epi_function = result.prox_expr.prox_function
                epi_function.epigraph = True

                return MatchResult(
                    True,
                    expression.prox_function(
                        epi_function, *(result.prox_expr.arg + [t_expr])),
                    result.constrs)

    return MatchResult(False)

# Conic transform (catch-all default)

def transform_cone(expr):
    obj, constrs = conic.transform_expr(expr)
    return MatchResult(True, None, [obj] + constrs)

# Used for both proximal/epigraph operators
BASE_RULES = [
    # Matrix
    prox_lambda_max,
    prox_neg_log_det,
    prox_norm_nuclear,
    prox_semidefinite,

    # Vector
    prox_max,
    prox_norm_2,
    prox_second_order_cone,
    prox_sum_largest,
    prox_total_variation_1d,

    # Elementwise
    prox_norm_1,
    prox_sum_exp,
    prox_sum_inv_pos,
    prox_sum_logistic,
    prox_sum_neg_entr,
    prox_sum_neg_log,

    # NOTE(mwytock): Maintain this order as deadzone specializes hinge
    prox_sum_deadzone,
    prox_sum_quantile,
    prox_sum_hinge,
]


PROX_RULES = [
    # Affine
    prox_sum_square,
    prox_zero,

    # Simple
    prox_constant,
    prox_affine,
]

PROX_RULES += BASE_RULES

PROX_RULES += [
    epigraph,
    prox_non_negative,

    # Lowest priority, transform to cone problem
    transform_cone,
]

def transform_expr(expr):
    log_debug_expr("transform_expr", expr)
    for rule in PROX_RULES:
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
