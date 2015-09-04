
from distopt import expression_pb2
from distopt import prox_pb2
from distopt.expression_pb2 import *

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
                                  Expression.P_NORM):
        c = "p: " + str(expr.p)

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

def prox_problem_str(problem):
    s = ""
    for f in problem.prox_function:
        s += prox_pb2.ProxFunction.Function.Name(f.function)
        s += " (alpha: " + str(f.alpha) + ")\n"
        for arg in f.arg:
            s += expr_str(arg) + "\n"

        if f.HasField("affine"):
            s += "Affine:\n"
            s += expr_str(f.affine) + "\n"

        if f.HasField("regularization"):
            s += "Regularization:\n"
            s += expr_str(f.regularization) + "\n"

        s += "\n"

    if problem.consensus_variable:
        s += "Consensus variables:\n"
        for cv in problem.consensus_variable:
            s += "%s, (num_instances: %d)\n" % (
                cv.variable_id, cv.num_instances)
        s += "\n"

    for constr in problem.equality_constraint:
        s += "Equality constraint:\n"
        s += expr_str(constr) + "\n"
        s += "\n"

    return s

# Mutators

def rename_var(old_id, new_id, expr):
    if (expr.expression_type == Expression.VARIABLE and
        expr.variable.variable_id == old_id):
        expr.variable.variable_id = new_id

    for arg in expr.arg:
        rename_var(old_id, new_id, arg)

# Accessors

def scalar_constant(expr):
    if len(expr.arg) == 0:
        assert expr.expression_type == Expression.CONSTANT
        assert dimension(expr) == 1
        return expr.constant.scalar

    assert False, "not implemented"

def dimension(expr):
    return expr.size.dim[0]*expr.size.dim[1]

def expr_vars(expr):
    retval = {}
    if expr.expression_type == Expression.VARIABLE:
        retval[expr.variable.variable_id] = expr
    else:
        for arg in expr.arg:
            retval.update(expr_vars(arg))
    return retval

# Constructors

def add(*args):
    assert len(args) > 0
    for i in range(len(args),1):
        assert args[0].size == args[i].size

    return Expression(
        expression_type=Expression.ADD,
        arg=args,
        size=args[0].size)

def multiply(*args):
    assert len(args) == 2
    assert args[0].size.dim[1] == args[1].size.dim[0]

    return Expression(
        expression_type=Expression.MULTIPLY,
        arg=args,
        size=Size(dim=[args[0].size.dim[0], args[1].size.dim[1]]))

def hstack(*args):
    e = Expression(expression_type=Expression.HSTACK)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
        else:
            assert e.size.dim[0] == arg.size.dim[0]
            e.size.dim[1] += arg.size.dim[1]

        e.arg.add().CopyFrom(arg)

    return e

def vstack(*args):
    e = Expression(expression_type=Expression.VSTACK)

    for i, arg in enumerate(args):
        if i == 0:
            e.size.dim.extend(arg.size.dim)
        else:
            assert e.size.dim[1] == arg.size.dim[1]
            e.size.dim[0] += arg.size.dim[0]

        e.arg.add().CopyFrom(arg)

    return e

def reshape(arg, m, n):
    assert m*n == dimension(arg)

    return Expression(
        expression_type=Expression.RESHAPE,
        arg=[arg],
        size=Size(dim=[m,n]))

def negate(arg):
    return Expression(
        expression_type=Expression.NEGATE,
        arg=[arg],
        size=arg.size)

def variable(m, n, variable_id):
    return Expression(
        expression_type=Expression.VARIABLE,
        size=Size(dim=[m, n]),
        variable=Variable(variable_id=variable_id))

def constant(m, n, scalar=0):
    return Expression(
        expression_type=Expression.CONSTANT,
        size=Size(dim=[m, n]),
        constant=Constant(scalar=scalar))
