"""Make sure objective terms are separable."""

from collections import defaultdict

from epsilon import error
from epsilon.expression_pb2 import Expression
from epsilon.expression import *

class SeparateError(error.ProblemError):
    pass

def expr_var_instances(expr):
    retval = defaultdict(list)
    if expr.expression_type == Expression.VARIABLE:
        retval[expr.variable.variable_id].append(expr)
    for arg in expr.arg:
        retval.update(expr_var_instances(arg))
    return retval

def parent_map(expr):
    retval = {}
    for arg in expr.arg:
        retval[id(arg)] = expr
        retval.update(parent_map(arg))
    return retval

def separate_vars(problem):
    parent = parent_map(problem.objective)

    var_function_groups = defaultdict(list)
    for f in problem.objective.arg:
        for var_id, var_instances in expr_var_instances(f).iteritems():
            var_function_groups[var_id].append(var_instances)

    for var_id, f_groups in var_function_groups.iteritems():
        for i, f_group in enumerate(f_groups[1:]):
            new_var_id = "separate:%s:%d" % (var_id, i)
            if (len(f_group) == 1 and
                parent[id(f_group[0])].expression_type == Expression.INDEX):
                # Special case, variable appears with INDEX
                # TODO(mwytock): Fix this so its not dependent on the ordering
                # of the functions
                index_var = parent[id(f_group[0])]
                m, n = index_var.size.dim
                new_var = variable(m, n, new_var_id)
                problem.constraint.extend(
                    [equality_constraint(new_var, index_var)])
                index_var.CopyFrom(new_var)

            else:
                # General case
                old_var = Expression()
                old_var.CopyFrom(f_group[0])
                for var_instance in f_group:
                    var_instance.variable.variable_id = new_var_id
                problem.constraint.extend(
                    [equality_constraint(old_var, f_group[0])])

def transform(problem):
    if problem.objective.expression_type != Expression.ADD:
        raise SeparateError("Objective root expression is not add", problem)

    separate_vars(problem)
    return problem
