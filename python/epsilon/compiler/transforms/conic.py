"""Conic transforms for non-linear functions."""

from epsilon import expression
from epsilon.compiler.transforms.transform_util import *
from epsilon.expression_pb2 import Curvature, Expression


def transform_abs(expr, args):
    x = only_arg(expr)
    t = epi_var(expr, "abs")
    return t, [expression.leq_constraint(x, t),
               expression.leq_constraint(-x, t)]

def transform_max_elemwise(expr, args):
    t = epi_var(expr, "max_elemwise")
    return t, [expression.leq_constraint(x, t) for x in args]

def transform_min_elemwise(expr, args):
    t = epi_var(expr, "min_elemwise")
    return t, [expression.leq_constraint(t, x) for x in args]

def transform_soc_elemwise(expr, args):
    t = epi_var(expr, "soc_elemwise")
    return t, [expression.soc_elemwise_constraint(t, x) for x in args]

def transform_quad_over_lin(expr, args):
    validate_args(expr, 2)
    x, y = expr.arg
    validate_size(y, (1,1))
    t = epi_var(expr, "qol", size=(1,1))
    return t, [
        expression.soc_constraint(
            expression.add(y, t),
            expression.add(y, expression.negate(t)),
            expression.mulitply(expression.constant(1, 1, scalar=2), x)),
        expression.leq_constraint(0, y)]

def transform_expr(expr):
    if expr.curvature.curvature_type in (Curvature.AFFINE, Curvature.CONSTANT):
        return expr, []
    else:
        args = []
        args_constr = []
        for arg in expr.arg:
            arg, arg_constr = transform_expr(arg)
            args.append(expr)
            args_constr += arg_constr

        f_name = "transform_" + Expression.Type.Name(expr.expression_type).lower()
        if f_name not in globals():
            raise TransformError("No conic transform", expr)
        expr, constr = globals()[f_name](expr, args)
        return expression.add(*([expr] + args_constr + constr))
