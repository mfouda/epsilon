"""Analyze the problem in sum-of-prox form and combine/split terms."""

from collections import defaultdict
from collections import namedtuple

from epsilon.compiler import validate
from epsilon.compiler.problem_graph import *
from epsilon.expression import *
from epsilon.expression_util import fp_expr
from epsilon.expression_pb2 import Expression

def is_affine(f):
    return f.expr.curvature.curvature_type == Curvature.AFFINE

def has_data_constant(expr):
    if (expr.expression_type == Expression.CONSTANT and
        expr.constant.data_location != ""):
        return True

    for arg in expr.arg:
        if has_data_constant(arg):
            return True

    return False

def prox_op(expr):
    if expr.expression_type == Expression.ADD:
        return prox_op(expr.arg[0])
    if expr.expression_type == Expression.MULTIPLY:
        return prox_op(expr.arg[1])
    return expr.proximal_operator.name

def is_prox_friendly_constraint(graph, f):
    """Returns true if f represents a prox-friendly equality constraint.

    In other words, one that can be treated as a constraint without interfering
    with the proximal operators for the other objective terms."""

    if not is_equality_indicator(f):
        return False

    for f_var in graph.edges_by_function[f]:
        edges = graph.edges_by_variable[f_var.variable]
        if len(edges) > 1 and not f_var.curvature.scalar_multiple:
            return False

    return True

def max_overlap_function(graph, f):
    """Return the objective term with maximum overlap in variables."""

    def variables(g):
        return set(g_var.variable for g_var in graph.edges_by_function[g])
    variables_f = variables(f)
    def overlap(g):
        return len(variables(g).intersection(variables_f))

    h = max((g for g in graph.obj_terms if g != f), key=overlap)

    # Only return a function if there is some overlap
    if overlap(h):
        return h

def separate_var(f_var):
    variable_id = "separate:%s:%s" % (
        f_var.variable, fp_expr(f_var.function.expr))
    return Expression(
        expression_type=Expression.VARIABLE,
        variable=Variable(variable_id=variable_id),
        size=f_var.instances[0].size)

def combine_affine_functions(graph):
    """Combine affine functions with other objective terms."""
    for f in graph.obj_terms:
        if not is_affine(f):
            continue

        g = max_overlap_function(graph, f)
        if not g:
            continue

        graph.remove_function(f)
        graph.remove_function(g)

        # NOTE(mwytock): The non-affine function must go
        # first. Fixing/maintaining this should likely go in a normalize step.
        graph.add_function(Function(add(g.expr, f.expr), constraint=False))

def move_equality_indicators(graph):
    """Move certain equality indicators from objective to constraints."""
    for function in graph.obj_terms:
        if (is_prox_friendly_constraint(graph, function)):
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

        skipped_one = False
        for f_var in reversed(f_vars):
            # Skip first one, rename the rest
            if not f_var.has_linops() and not skipped_one:
                skipped_one = True
                continue

            graph.remove_edge(f_var)

            new_var_id = "separate:%s:%s" % (
                f_var.variable, fp_expr(f_var.function.expr))
            old_var, new_var = f_var.replace_variable(new_var_id)
            graph.add_function(Function(equality_constraint(old_var, new_var),
                                        constraint=True))

            graph.add_edge(f_var)

def add_zero_prox(graph):
    """Add f(x) = 0 term for variables only appearing in constraints."""

    for var in graph.variables:
        f_vars = [f_var for f_var in graph.edges_by_variable[var]
                  if not f_var.function.constraint]

        if f_vars:
            continue

        # Create the zero function and add this variable to it
        # TODO(mwytock): ProblemGraph interface should probably be a bit better
        # for this use case.
        var_expr = graph.edges_by_variable[var][0].instances[0]
        f_expr = zero(var_expr)
        f_expr.proximal_operator.name = "ZeroProx"
        f = Function(f_expr, constraint=False)
        f_var = FunctionVariable(f, var, f_expr.arg[0])
        graph.add_function(f)
        graph.add_edge(f_var)

GRAPH_TRANSFORMS = [
    move_equality_indicators,
    combine_affine_functions,
    separate_objective_terms,
    add_zero_prox,
]

def transform(problem):
    graph = ProblemGraph(problem)
    for f in GRAPH_TRANSFORMS:
        f(graph)
    return graph.problem()
