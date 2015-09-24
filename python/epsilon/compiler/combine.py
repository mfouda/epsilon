"""Analyze the problem in sum-of-prox form and combine/split terms."""

from collections import defaultdict
from collections import namedtuple

from epsilon.compiler import validate
from epsilon.compiler import attributes
from epsilon.compiler.problem_graph import Function, ProblemGraph
from epsilon.expression import *
from epsilon.expression_pb2 import Expression

def is_affine(f):
    return f.expr.curvature.curvature_type == Curvature.AFFINE

def is_equality_indicator(f):
    return (not f.constraint and
            f.expr.expression_type == Expression.INDICATOR and
            f.expr.expression_type == Cone.ZERO)

def is_sparse_equality_constraint(f):
    # TODO(mwytock): Walk the tree check for constant terms
    return False

def is_prox_friendly_constraint(graph, f):
    """Returns true if f represents a prox-friendly equality constraint,
    i.e. one that can be treated as a constraint without interfering with the
    proximal operators for the other objective terms."""
    if not is_equality_indicator(f):
        return False

    for f_var in graph.edges_by_function[f]:
        edges = graph.edges_by_variable[f_var.variable]
        if len(edges) > 1 and not f_var.affine_curvature.scalar_multiple:
            return False

    return True

def max_overlap(graph, f):
    """Return the objective term with maximum overlap in variables."""

    def variables(g):
        return set(g_var.variable for g_var in graph.edges_by_function[g])
    variables_f = variables(f)
    def overlap(g):
        return len(variables(g).intersection(variables_f))

    return max((g for g in graph.functions if g != f), key=overlap)

def separate_var(f_var):
    variable_id = "separate:%s:%s" % (
        f_var.variable, fp_expr(f_var.function.expr))
    return Expression(
        expression_type=Expression.VARIABLE,
        variable=Variable(variable_id=variable_id),
        size=f_var.var_expr.size)

def combine_affine_functions(graph):
    """Combine affine functions with other objective terms."""
    for function in graph.functions:
        if not is_affine(function):
            continue

        other = max_overlap(graph, function)
        if not other:
            continue

        graph.remove_function(other)
        graph.remove_function(affine)
        graph.add_function(Function(add(other, affine), constraint=True))

def move_equality_indicators(graph):
    """Move certain equality indicators from objective to constraints."""
    for function in graph.functions:
        if (is_sparse_equality_constraint(function) or
            is_prox_friendly_constraint(graph, function)):
            function.constraint = True

def separate_objective_terms(graph):
    """Add variable copies to make functions separable.

    This applies to objective functions only and we dont need to modify the
    first occurence.
   """
    for var in graph.variables:
        # Exclude constraint terms
        f_vars = [f_var for f_var in graph.edges_by_variable[var]
                  if not f_var.function.constraint]

        # Skip first one, rename the rest
        for f_var in f_vars[1:]:
            new_var_expr = separate_var(f_var)
            graph.add_function(
                Function(equality_constraint(f_var.var_expr, new_var_expr),
                         constraint=True))

            graph.remove_edge(f_var)
            f_var.replace_variable(new_var_expr)
            graph.add_edge(f_var)

GRAPH_TRANSFORMS = [
    combine_affine_functions,
    move_equality_indicators,
    separate_objective_terms,
]

def transform(problem):
    graph = ProblemGraph(problem)
    for f in GRAPH_TRANSFORMS:
        f(graph)
    return graph.problem()
