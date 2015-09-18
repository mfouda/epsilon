"""Finalize various things before passing to solver."""

from epsilon.expression import *

def final_indicator(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO
        and len(expr.arg) > 1):
        add_expr = add(expr.arg[0], *(negate(arg) for arg in expr.arg[1:]))
        expr.ClearField("arg")
        expr.arg.add().CopyFrom(add_expr)

FINAL_RULES = [f for name, f in locals().items() if name.startswith("final_")]
def transform_expr(expr):
    for arg in expr.arg:
        transform_expr(arg)

    for rule in FINAL_RULES:
        rule(expr)

def transform(input):
    transform_expr(input.objective)
    for constr in input.constraint:
        transform_expr(constr)
    return input
