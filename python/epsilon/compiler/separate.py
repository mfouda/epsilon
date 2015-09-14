"""Make sure objective terms are separable."""

from collections import defaultdict

from epsilon.compiler import compiler_error
from epsilon.expression_pb2 import Expression
from epsilon.expression import *

class SeparateError(compiler_error.CompilerError):
    pass

def rename_var(old_id, new_id, expr):
    if (expr.expression_type == Expression.VARIABLE and
        expr.variable.variable_id == old_id):
        expr.variable.variable_id = new_id

    for arg in expr.arg:
        rename_var(old_id, new_id, arg)

def transform(problem):
    """If two prox functions refer to the same variable, add a copy."""

    if problem.objective.expression_type != Expression.ADD:
        raise SeparateError("Objective root expression is not add", problem)

    var_function_map = defaultdict(list)
    orig_vars = {}
    for f in problem.objective.arg:
        f_vars = expr_vars(f)
        orig_vars.update(f_vars)
        for var_id in f_vars:
            var_function_map[var_id].append(f)

    for var_id, fs in var_function_map.iteritems():
        # Enumerate the functions backwards so that the last prox function keeps
        # the original variable id.
        for i, f in enumerate(fs[-2::-1]):
            new_var_id = "separate:%s:%d" % (var_id, i)
            old_var = orig_vars[var_id]
            new_var = Expression()
            new_var.CopyFrom(old_var)
            new_var.variable.variable_id = new_var_id

            problem.constraint.extend([equality_constraint(old_var, new_var)])
            rename_var(var_id, new_var_id, f)

    return problem
