"""Represents the bipartite graph between functions/variables.

Functions can either be objective terms or constraints and edges represent the
usage of a variable by a function constraint.

We wrap the expression tree form from expression_pb2 with thin Python objects
that allow for caching of computations and object-oriented model for
mutations.
"""

from collections import defaultdict

from epsilon import expression
from epsilon.compiler import attributes
from epsilon.compiler import validate
from epsilon.expression_pb2 import Expression, Problem, Cone

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
    def __init__(self, function, variable, instances):
        self.function = function
        self.variable = variable
        self.instances = instances

    def replace_variable(self, new_var_expr):
        self.variable = new_var_expr.variable.variable_id
        for instance in self.instances:
            instance.CopyFrom(new_var_expr)

def find_var_instances(expr):
    if expr.expression_type == Expression.VARIABLE:
        return {expr.variable.variable_id: [expr]}

    retval = defaultdict(list)
    for arg in expr.arg:
        for var_id, instances in find_var_instances(arg).iteritems():
            retval[var_id] += instances

    return retval

class ProblemGraph(object):
    def __init__(self, problem):
        self.edges_by_variable = defaultdict(list)
        self.edges_by_function = defaultdict(list)

        validate.check_sum_of_prox(problem)
        for f_expr in problem.objective.arg:
            self.add_function(Function(f_expr, constraint=False))
        for f_expr in problem.constraint:
            self.add_function(Function(f_expr, constraint=True))

    def problem(self):
        return Problem(
            objective=expression.add(*(f.expr for f in self.functions)),
            constraint=(f.expr for f in self.constraints))

    # Basic operations
    def remove_edge(self, f_var):
        self.edges_by_variable[f_var.variable].remove(f_var)
        self.edges_by_function[f_var.function].remove(f_var)

    def add_edge(self, f_var):
        self.edges_by_variable[f_var.variable].append(f_var)
        self.edges_by_function[f_var.function].append(f_var)

    def remove_function(self, f):
        for f_var in self.edges_by_function[f]:
            self.edges_by_variable[f_var.variable].remove(f_var)
        del self.edges_by_function[f]

    def add_function(self, f):
        for variable, instances in find_var_instances(f.expr).iteritems():
            self.add_edge(FunctionVariable(f, variable, instances))

        # Add curvature attributes for equality indicator functions
        if is_equality_indicator(f):
            var_curvature = attributes.compute_variable_curvature(f.expr.arg[0])
            for f_var in self.edges_by_function[f]:
                f_var.curvature = var_curvature[f_var.variable]

    # Accessors for nodes
    @property
    def functions(self):
        return [f for f in self.edges_by_function.keys() if not f.constraint]

    @property
    def constraints(self):
        return [f for f in self.edges_by_function.keys() if f.constraint]

    @property
    def variables(self):
        return self.edges_by_variable.keys()
