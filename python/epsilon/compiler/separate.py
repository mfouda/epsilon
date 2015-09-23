"""Make sure objective terms are separable."""

from collections import defaultdict
from collections import namedtuple

from epsilon.compiler import validate
from epsilon.compiler import attributes
from epsilon.expression import *
from epsilon.expression_pb2 import Expression

def separate_var(f, size, orig_var_id):
    var_id = "separate:" + orig_var_id + ":" + fp_expr(f)
    return Expression(
        expression_type=Expression.VARIABLE,
        variable=Variable(variable_id=var_id),
        size=size)

class FunctionVar(object):
    """Represents usage of a variable inside a single function."""

    def __init__(self):
        self.nonindex_vars = []
        self.index_vars = []
        self.scalar_equality = True
        self.f_expr = None
        self.var_id = None

    def make_copy(self):
        if not self.nonindex_vars and len(self.index_vars) == 1:
            # Special case for a single index_var
            index_var = self.index_vars[0]
            new_var = separate_var(
                self.f_expr, self.index_vars[0].size, self.var_id)
            old_var = Expression()
            old_var.CopyFrom(index_var)
            index_var.CopyFrom(new_var)
            return old_var, new_var
        else:
            assert not self.index_vars
            old_var = Expression()
            old_var.CopyFrom(self.nonindex_vars[0])
            new_var = separate_var(self.f_expr, old_var.size, self.var_id)
            for var in self.nonindex_vars:
                var.variable.variable_id = new_var.variable.variable_id
            return old_var, new_var

    def update(self, function_var):
        self.nonindex_vars += function_var.nonindex_vars
        self.index_vars += function_var.index_vars
        self.scalar_equality = (self.scalar_equality and function_var.scalar_equality)

def is_scalar_equality(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO):
        return True

    return attributes.is_scalar_expression(expr)

def function_vars(expr):
    retval = defaultdict(FunctionVar)

    if (expr.expression_type == Expression.INDEX and
        expr.arg[0].expression_type == Expression.VARIABLE):
        retval[expr.arg[0].variable.variable_id].index_vars.append(expr)

    elif expr.expression_type == Expression.VARIABLE:
        retval[expr.variable.variable_id].nonindex_vars.append(expr)

    else:
        for arg in expr.arg:
            for var_id, function_var in function_vars(arg).iteritems():
                retval[var_id].scalar_equality = is_scalar_equality(expr)
                retval[var_id].update(function_var)

    return retval

def separate_vars(problem):
    var_function_map = defaultdict(list)
    for f in problem.objective.arg:
        for var_id, function_var in function_vars(f).iteritems():
            function_var.var_id = var_id
            function_var.f_expr = f
            if not function_var.scalar_equality:
                var_function_map[var_id].append(function_var)

    for var_id, function_var_list in var_function_map.iteritems():
        for i, function_var in enumerate(function_var_list[:-1]):
            old_var, new_var = function_var.make_copy()
            problem.constraint.extend([equality_constraint(old_var, new_var)])

def transform(problem):
    validate.check_sum_of_prox(problem)
    separate_vars(problem)
    return problem
