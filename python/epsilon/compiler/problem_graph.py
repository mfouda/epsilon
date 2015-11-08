"""Represents the bipartite graph between functions/variables.

Functions can either be objective terms or constraints and edges represent the
usage of a variable by a function constraint.

We wrap the expression tree form from expression_pb2 with thin Python objects
that allow for caching of computations and object-oriented model for
mutations.
"""

from collections import defaultdict

from epsilon import expression
from epsilon import expression_util
from epsilon.compiler import validate
from epsilon.expression_pb2 import Expression, Problem, Cone
from epsilon import tree_format

def is_equality_indicator(f):
    return (f.expr.expression_type == Expression.INDICATOR and
            f.expr.cone.cone_type == Cone.ZERO)

class Function(object):
    """Function node."""
    def __init__(self, expr, constraint):
        self.expr = expr
        self.constraint = constraint

class FunctionVariable(object):
    """Edge connecting a variable and function."""
    def __init__(self, function, variable, instance):
        self.function = function
        self.variable = variable
        self.instances = [instance]
        self.instances_with_linops = [instance]

    def has_linops(self):
        for i, instance in enumerate(self.instances):
            if instance != self.instances_with_linops[i]:
                return True
            return False

    def combine(self, other):
        assert self.function == other.function
        assert self.variable == other.variable
        self.instances += other.instances
        self.instances_with_linops += other.instances_with_linops

    def replace_variable(self, new_var_id):
        for instance in self.instances_with_linops[1:]:
            if instance != self.instances_with_linops[0]:
                all_linops_equal = False
                break
        else:
            all_linops_equal = True

        if all_linops_equal:
            to_replace = self.instances_with_linops
        else:
            to_replace = self.instances

        old_var = Expression()
        old_var.CopyFrom(to_replace[0])
        m, n = old_var.size.dim

        new_var = expression.variable(m, n, new_var_id)
        for instance in to_replace:
            instance.CopyFrom(new_var)

        self.variable = new_var_id
        return old_var, new_var

def find_var_instances(f, expr):
    if expr.expression_type == Expression.VARIABLE:
        var = expr.variable.variable_id
        return {var: FunctionVariable(f, var, expr)}

    # TODO(mwytock): Generalize this to any kind of lin op and arbitrary chains.
    if (expr.expression_type == Expression.INDEX and
        expr.arg[0].expression_type == Expression.VARIABLE):
        retval = find_var_instances(f, expr.arg[0])
        for var_id, f_var in retval.iteritems():
            assert len(f_var.instances_with_linops) == 1
            f_var.instances_with_linops = [expr]
        return retval

    retval = {}
    for arg in expr.arg:
        for var_id, f_var in find_var_instances(f, arg).iteritems():
            if var_id in retval:
                retval[var_id].combine(f_var)
            else:
                retval[var_id] = f_var

    return retval

class ProblemGraph(object):
    def __init__(self, problem):
        self.functions = []
        self.edges_by_variable = defaultdict(list)
        self.edges_by_function = defaultdict(list)

        validate.check_sum_of_prox(problem)
        for f_expr in problem.objective.arg:
            self.add_function(Function(f_expr, constraint=False))
        for f_expr in problem.constraint:
            self.add_function(Function(f_expr, constraint=True))

    def problem(self):
        return Problem(
            objective=expression.add(*(f.expr for f in self.obj_terms)),
            constraint=[f.expr for f in self.constraints])

    # Basic operations
    def remove_edge(self, f_var):
        self.edges_by_variable[f_var.variable].remove(f_var)
        self.edges_by_function[f_var.function].remove(f_var)

    def add_edge(self, f_var):
        self.edges_by_variable[f_var.variable].append(f_var)
        self.edges_by_function[f_var.function].append(f_var)

    def remove_function(self, f):
        self.functions.remove(f)
        for f_var in self.edges_by_function[f]:
            self.edges_by_variable[f_var.variable].remove(f_var)
        del self.edges_by_function[f]

    def add_function(self, f):
        f_vars = find_var_instances(f, f.expr)
        if not f_vars:
            return

        self.functions.append(f)
        for f_var in f_vars.itervalues():
            self.add_edge(f_var)

        # Add curvature attributes for equality indicator functions
        if is_equality_indicator(f):
            var_curvature = expression_util.compute_variable_curvature(
                f.expr.arg[0])
            for f_var in self.edges_by_function[f]:
                f_var.curvature = var_curvature[f_var.variable]

    # Accessors for nodes
    @property
    def obj_terms(self):
        return [f for f in self.functions if not f.constraint]

    @property
    def constraints(self):
        return [f for f in self.functions if f.constraint]

    @property
    def variables(self):
        return self.edges_by_variable.keys()
