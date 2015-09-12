
from epsilon.expression_pb2 import *

def _node_contents_str(expr):
    c = ""

    if expr.expression_type == Expression.CONSTANT:
        if expr.constant.data_location:
            c = "data_location: " + expr.constant.data_location
        else:
            c = "scalar: " + str(expr.constant.scalar)
    elif expr.expression_type == Expression.VARIABLE:
        c = "variable_id: " + expr.variable.variable_id
    elif expr.expression_type == Expression.INDEX:
        c = "key: [" + ", " .join([
            "%d:%d%s" % (k.start, k.stop, "" if k.step == 1 else ":%d" % k.step)
            for k in expr.key]) + "]"
    elif expr.expression_type in (Expression.POWER,
                                  Expression.NORM_P):
        c = "p: " + str(expr.p)
    elif expr.expression_type == Expression.INDICATOR:
        c = "cone: " + Cone.Type.Name(expr.cone.cone_type)

    return "(%s)" % c if c else ""

def _node_size_str(expr):
    return "%-10s" % ("(" + ", ".join(str(d) for d in expr.size.dim) + ")",)

def _node_str(expr, pre):
    return (_node_size_str(expr) + "\t" + pre +
            Expression.Type.Name(expr.expression_type) + " " +
            _node_contents_str(expr))

def expr_str(expr, pre=""):
    return "\n".join(
        [_node_str(expr, pre)] +
        [expr_str(a, pre=pre + "  ") for a in expr.arg])

def problem_str(problem):
    s = "Objective:\n" + expr_str(problem.objective)
    for constr in problem.constraint:
        s += "\nConstraint:\n"
        for arg in constr.arg:
            s += expr_str(arg) + "\n"
    return s
