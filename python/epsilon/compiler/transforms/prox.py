"""Transform a problem to prox-affine form.

TODO(mwytock): Clean up the interaction between expression matching and args
extraction.
"""

import logging
from collections import namedtuple

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
        if arg.dcp_props.affine:
            args.append(linear.transform_expr(arg))
            continue

        logging.debug("not affine, adding epigraph")
        t, epi_f = epi_transform(arg, "affine")
        args.append(t)
        constr.append(epi_f)

    return args, constr

def convert_arg(expr, affine_check):
    if expr.dcp_props.affine:
        expr_linear = linear.transform_expr(expr)
        if affine_check(expr_linear):
            return expr_linear, []

    t, epi_f = epi_transform(expr, "convert_arg")
    return t, [epi_f]

def diagonal_args(expr):
    args = []
    constr = []

    for arg in expr.arg:
        if arg.dcp_props.affine:
            arg_linear = linear.transform_expr(arg)
            if arg_linear.affine_props.diagonal:
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
        if arg.dcp_props.affine:
            arg_linear = linear.transform_expr(arg)
            if arg_linear.affine_props.scalar:
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

def expr_args(convert):
    def args(expr):
        output_args = []
        constr = []
        for input_arg in expr.arg:
            arg_i, constr_i = convert(input_arg)
            output_args.append(arg_i)
            constr += constr_i
        return output_args, constr
    return args

def soc_prox_args(expr):
    return epigraph_args(
        lambda e: convert_arg(e, lambda e: e.affine_props.scalar))(expr)

def create_prox(prox_function_type, **kwargs):
    def create(expr):
        kwargs["prox_function_type"] = prox_function_type
        return ProxFunction(**kwargs)
    return create

def create_matrix_prox(prox_function_type):
    def create(expr):
        return ProxFunction(
            prox_function_type=prox_function_type,
            m=dim(expr.arg[0], 0),
            n=dim(expr.arg[0], 1))
    return create

def args_default(expr):
    return expr.arg

def args_epigraph(expr):
    f_expr, t_expr = get_epigraph(expr)
    return [t_expr] + list(f_expr.arg)

def create_second_order_cone(expr, f_args=args_default):
    args = f_args(expr)
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
        lambda e: create_second_order_cone(e, epigraph_args)),
]

# Linear cone rules
RULES += [
    ProxRule(lambda e: e.dcp_props.constant, affine_args,
             create_prox(Prox.CONSTANT)),
    ProxRule(lambda e: e.dcp_props.affine, affine_args,
             create_prox(Prox.AFFINE)),
    ProxRule(match_indicator(Cone.ZERO), least_squares_args,
             create_prox(Prox.ZERO)),
    ProxRule(match_indicator(Cone.NON_NEGATIVE), diagonal_args,
             create_prox(Prox.NON_NEGATIVE)),
    ProxRule(match_indicator(Cone.SECOND_ORDER), scalar_args,
             create_second_order_cone),
    ProxRule(match_indicator(Cone.SEMIDEFINITE), scalar_args,
             create_matrix_prox(Prox.SEMIDEFINITE)),
]

def transform_prox_expr(rule, expr):
    args, constrs = rule.convert_args(expr)
    prox = rule.create(expr)
    yield expression.prox_function(prox, *args)
    for constr in constrs:
        for f_expr in transform_expr(constr):
            yield f_expr

def transform_cone_expr(expr):
    obj, constrs = conic.transform_expr(expr)
    for expr in [obj] + constrs:
        for f_expr in transform_expr(expr):
            yield f_expr

def transform_expr(expr):
    log_debug_expr("transform_expr", expr)
    for rule in RULES:
        if rule.match(expr):
            for f_expr in transform_prox_expr(rule, expr):
                yield f_expr
            break
    else:
        # TODO(mwytock): Make this a rule with lowest priority
        for f_expr in transform_cone_expr(expr):
            yield f_expr

def transform_problem(problem):
    f_exprs = list(transform_expr(problem.objective))
    for constr in problem.constraint:
        f_exprs += list(transform_expr(constr))
    return expression.Problem(objective=expression.add(*f_exprs))
