
from epsilon.expression_pb2 import *

def key_str(expr):
    return "[" + ", " .join([
        "%d:%d%s" % (k.start, k.stop, "" if k.step == 1 else ":%d" % k.step)
        for k in expr.key]) + "]"

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

    if expr.proximal_operator.name != "":
        c += ["prox: " + expr.proximal_operator.name]

    return "(" + ", ".join(c) + ")" if c else ""

def _node_size_str(expr):
    return "%-10s" % ("(" + ", ".join(str(d) for d in expr.size.dim) + ")",)

def node_str(expr, pre=""):
    return (_node_size_str(expr) + "\t" + pre +
            Expression.Type.Name(expr.expression_type) + " " +
            node_contents_str(expr))

def expr_str(expr, pre=""):
    return "\n".join(
        [node_str(expr, pre)] +
        [expr_str(a, pre=pre + "  ") for a in expr.arg])

def problem_str(problem):
    s = "Objective:\n" + expr_str(problem.objective)
    for constr in problem.constraint:
        s += "\nConstraint:\n" + expr_str(constr)
    return s
