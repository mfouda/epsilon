"""Transform a problem to prox-affine form."""

import logging
from collections import namedtuple

from epsilon import affine
from epsilon import dcp
from epsilon import expression
from epsilon import tree_format
from epsilon.compiler.transforms import conic
from epsilon.compiler.transforms import linear
from epsilon.compiler.transforms.transform_util import *
from epsilon.expression_pb2 import Cone, Expression, ProxFunction, Problem

ProxRule = namedtuple("ProxRule", ["match", "convert_args", "create"])

RULES = []

# Shorthand convenience
Prox = ProxFunction

def affine_args(expr):
    return [linear.transform_expr(expr)], []

def least_squares_args(expr):
    args = []
    constr = []

    for arg in expr.arg:
        if dcp.is_affine(arg):
            args.append(linear.transform_expr(arg))
            continue

        logging.debug("not affine, adding epigraph")
        t, epi_f = epi_transform(arg, "affine")
        args.append(t)
        constr.append(epi_f)

    return args, constr

def convert_diagonal(expr):
    if dcp.is_affine(expr):
        expr_linear = linear.transform_expr(expr)
        if affine.is_diagonal(expr_linear):
            return expr_linear, []

    logging.debug("not diagonal, adding epigraph")
    t, epi_f = epi_transform(expr, "diagonal")
    return t, [epi_f]

def diagonal_args(expr):
    args = []
    constr = []

    for arg in expr.arg:
        if dcp.is_affine(arg):
            arg_linear = linear.transform_expr(arg)
            if affine.is_diagonal(arg_linear):
                args.append(arg_linear)
                continue

        logging.debug("not diagonal, adding epigraph")
        t, epi_f = epi_transform(arg, "diagonal")
        args.append(t)
        constr.append(epi_f)

    return args, constr

def scalar_args(expr):
    args = []
    constr = []

    for arg in expr.arg:
        if dcp.is_affine(arg):
            arg_linear = linear.transform_expr(arg)
            if affine.is_scalar(arg_linear):
                args.append(arg_linear)
                continue

        logging.debug("not scalar, adding epigraph")
        t, epi_f = epi_transform(arg, "scalar")
        args.append(t)
        constr.append(epi_f)

    return args, constr

def epigraph_args(convert):
    def args(expr):
        f_expr, t_expr = get_epigraph(expr)
        output_args = []
        constr = []
        for input_arg in [t_expr] + [a for a in f_expr.arg]:
            arg_i, constr_i = convert(input_arg)
            output_args.append(arg_i)
            constr += constr_i
        return output_args, constr
    return args

def soc_prox_args(expr):
    args, constr = epigraph_args(convert_diagonal)(expr)
    # Make argument a row vector
    x_args = [expression.reshape(x, 1, dim(x)) for x in args[1:]]
    return [args[0]] + x_args, constr

def create(prox_function_type, **kwargs):
    def create(expr, args):
        kwargs["prox_function_type"] = prox_function_type
        return ProxFunction(**kwargs)
    return create

def create_second_order_cone(expr, args):
    return ProxFunction(
        prox_function_type=Prox.SECOND_ORDER_CONE,
        m=dim(args[1], 0),
        n=dim(args[1], 1))

def match_epigraph(f_match):
    def match(expr):
        f_expr, t_expr = get_epigraph(expr)
        if not f_expr:
            return False
        return f_match(f_expr)
    return match

def match_indicator(cone_type):
    def match(expr):
        return (expr.expression_type == Expression.INDICATOR and
                expr.cone.cone_type == cone_type)
    return match

# Proximal operator rules
RULES += [
    ProxRule(
        match_epigraph(
            lambda e: e.expression_type == Expression.NORM_P and e.p == 2),
        soc_prox_args,
        create_second_order_cone),
]

# Linear cone rules
RULES += [
    ProxRule(dcp.is_affine, affine_args, create(Prox.AFFINE)),
    ProxRule(match_indicator(Cone.ZERO), least_squares_args, create(Prox.ZERO)),
    ProxRule(match_indicator(Cone.NON_NEGATIVE), diagonal_args,
             create(Prox.NON_NEGATIVE)),
    ProxRule(match_indicator(Cone.SECOND_ORDER), scalar_args,
             create_second_order_cone),
]

def merge_add(a, b):
    args = []
    args += a.arg if a.expression_type == Expression.ADD else [a]
    args += b.arg if b.expression_type == Expression.ADD else [b]
    return expression.add(*args)

def transform_prox_expr(rule, expr):
    logging.debug("transform_prox_expr:\n%s", tree_format.format_expr(expr))
    args, constrs = rule.convert_args(expr)
    prox = rule.create(expr, args)
    expr = expression.add(expression.prox_function(prox, *args))
    for constr in constrs:
        expr = merge_add(expr, transform_expr(constr))
    logging.debug(
        "transform_prox_expr done:\n%s", tree_format.format_expr(expr))
    return expr

def transform_cone_expr(expr):
    logging.debug("transform_cone_expr:\n%s", tree_format.format_expr(expr))
    expr, constrs = conic.transform_expr(expr)
    expr = transform_expr(expr)
    for constr in constrs:
        expr = merge_add(expr, transform_expr(constr))
    logging.debug(
        "transform_cone_expr done:\n%s", tree_format.format_expr(expr))
    return expr

def transform_expr(expr):
    logging.debug("transform_expr:\n%s", tree_format.format_expr(expr))
    for rule in RULES:
        if rule.match(expr):
            return transform_prox_expr(rule, expr)
    return transform_cone_expr(expr)

def transform_problem(problem):
    expr = transform_expr(problem.objective)
    for constr in problem.constraint:
        expr = merge_add(expr, transform_expr(constr))
    return Problem(objective=expr)
