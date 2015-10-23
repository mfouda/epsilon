
from epsilon.expression_pb2 import *

def key_str(expr):
    return "[" + ", " .join([
        "%d:%d%s" % (k.start, k.stop, "" if k.step == 1 else ":%d" % k.step)
        for k in expr.key]) + "]"

def format_linear_map(linear_map):
    name = LinearMap.Type.Name(linear_map.linear_map_type).lower()
    if linear_map.arg:
        args = [format_linear_map(arg) for arg in linear_map.arg]
    else:
        args = [str(linear_map.m), str(linear_map.n)]
    return name + "(" + ", ".join(args) + ")"

def node_contents_str(expr):
    c = []

    if expr.expression_type == Expression.CONSTANT:
        if not expr.constant.data_location:
            c += ["scalar: " + str(expr.constant.scalar)]
    elif expr.expression_type == Expression.VARIABLE:
        c += ["variable_id: " + expr.variable.variable_id]
    elif expr.expression_type == Expression.INDEX:
        c += ["key: " + key_str(expr)]
    elif expr.expression_type in (Expression.POWER,
                                  Expression.NORM_P):
        c += ["p: " + str(expr.p)]
    elif expr.expression_type == Expression.SUM_LARGEST:
        c += ["k: " + str(expr.k)]
    elif expr.expression_type == Expression.INDICATOR:
        c += ["cone: " + Cone.Type.Name(expr.cone.cone_type)]
    elif expr.expression_type == Expression.LINEAR_MAP:
        c += [format_linear_map(expr.linear_map)]

    if expr.proximal_operator.name != "":
        c += ["prox: " + expr.proximal_operator.name]

    return "(" + ", ".join(c) + ")" if c else ""

def _node_size_str(expr):
    return "%-10s" % ("(" + ", ".join(str(d) for d in expr.size.dim) + ")",)

def node_str(expr, pre=""):
    return (_node_size_str(expr) + "\t" + pre +
            Expression.Type.Name(expr.expression_type) + " " +
            node_contents_str(expr))

def format_expr(expr, pre=""):
    return "\n".join(
        [node_str(expr, pre)] +
        [format_expr(a, pre=pre + "  ") for a in expr.arg])

def format_problem(problem):
    s = "Objective:\n" + format_expr(problem.objective)
    for constr in problem.constraint:
        s += "\nConstraint:\n" + format_expr(constr)
    return s
