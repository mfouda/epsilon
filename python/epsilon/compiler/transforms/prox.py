"""Transform a problem to prox-affine form."""

from collections import namedtuple

from epsilon import expression
from epsilon.compiler.transforms import conic
from epsilon.expression import Cone, Expression, ProxFunction

ProxRule = namedtuple("ProxRule", ["match", "convert_args", "create"])

RULES = []

# Shorthand convenience
Prox = ProxFunction

def any_args(expr):
    return expr.arg

def diagonal_args(expr):
    # TODO(mwytock): Verify elementwise
    return expr.arg

def scalar_args(expr):
    # TODO(mwytock): Verify scalar
    return expr.arg

def cone_rule(cone_type, prox_function_type, convert_args, elementwise=False):
    def match(expr):
        return (expr.expression_type == Expression.INDICATOR and
                expr.cone.cone_type == cone_type)

    def create(args):
        return ProxFunction(
            prox_function_type=prox_function_type,
            elementwise=elementwise)

    return ProxRule(match, convert_args, create)

# Cone rules
RULES += [
    cone_rule(Cone.ZERO, Prox.ZERO, any_args),
    cone_rule(Cone.NON_NEGATIVE, Prox.NON_NEGATIVE, diagonal_args),
    cone_rule(Cone.SECOND_ORDER, Prox.SECOND_ORDER_CONE, scalar_args),
    cone_rule(Cone.SECOND_ORDER_ELEMENTWISE,
              Prox.SECOND_ORDER_CONE,
              diagonal_args,
              elementwise=True)
]

def merge_add(a, b):
    args = []
    args += a.arg if a.expression_type == Expression.ADD else [a]
    args += b.arg if b.expression_type == Expression.ADD else [b]
    return expression.add(*args)

def transform_prox_expr(rule, expr):
    args, indicators = rule.convert_args(expr)
    expr = expression.prox_function(rule.create(), *args)
    for indicator in indicators:
        expr = merge_add(expr, transform_expr(indicator))
    return expr

def transform_expr(expr):
    for rule in RULES:
        if rule.match(expr):
            return transform_prox_expr(rule, expr)
    return conic.transform_expr(expr)

def transform_problem(problem):
    expr = transform_expr(problem.objective)
    for constr in problem.constraint:
        expr = merge_add(expr, transform_expr(constr))
    return Problem(objective=expr)
