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


def create(prox_function_type, **kwargs):
    def create():
        kwargs["prox_function_type"] = prox_function_type
        return ProxFunction(**kwargs)
    return create

def match_indicator(cone_type):
    def match(expr):
        return (expr.expression_type == Expression.INDICATOR and
                expr.cone.cone_type == cone_type)
    return match

# Linear cone rules
RULES += [
    ProxRule(dcp.is_affine, affine_args, create(Prox.AFFINE)),
    ProxRule(match_indicator(Cone.ZERO), least_squares_args, create(Prox.ZERO)),
    ProxRule(match_indicator(Cone.NON_NEGATIVE), diagonal_args,
             create(Prox.NON_NEGATIVE)),
    ProxRule(match_indicator(Cone.SECOND_ORDER), scalar_args,
             create(Prox.SECOND_ORDER_CONE)),
]

def merge_add(a, b):
    args = []
    args += a.arg if a.expression_type == Expression.ADD else [a]
    args += b.arg if b.expression_type == Expression.ADD else [b]
    return expression.add(*args)

def transform_prox_expr(rule, expr):
    logging.debug("transform_prox_expr:\n%s", tree_format.format_expr(expr))
    args, constrs = rule.convert_args(expr)
    f = rule.create()
    expr = expression.add(expression.prox_function(rule.create(), *args))
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
