"""Analyze the problem in sum-of-prox form and combine/split terms."""

from collections import defaultdict
from collections import namedtuple

from epsilon import expression_str
from epsilon.compiler import attributes
from epsilon.compiler import validate
from epsilon.compiler.problem_graph import *
from epsilon.expression import *
from epsilon.expression_pb2 import Expression

def is_affine(f):
    return f.expr.curvature.curvature_type == Curvature.AFFINE

def has_constant(expr):
    if expr.expression_type == Expression.CONSTANT:
        return True

    for arg in expr.arg:
        if has_constant(arg):
            return True

    return False

def is_sparse_equality_constraint(f):
    if not is_equality_indicator(f):
        return False
    return not has_constant(f.expr)

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

    return max((g for g in graph.functions if g != f), key=overlap)

def separate_var(f_var):
    return Expression(
        expression_type=Expression.VARIABLE,
        variable=Variable(variable_id=variable_id),
        size=f_var.instances[0].size)

def combine_affine_functions(graph):
    """Combine affine functions with other objective terms."""
    for f in graph.functions:
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
            old_var = f_var.instances[0]
            m, n = old_var.size.dim
            new_var = variable(m, n, ("separate:" +
                                      f_var.variable + ":" +
                                      fp_expr(f_var.function.expr)))

            graph.add_function(
                Function(equality_constraint(old_var, new_var),
                         constraint=True))

            graph.remove_edge(f_var)
            f_var.replace_variable(new_var)
            graph.add_edge(f_var)


def find_index_var_instances(expr):
    if (expr.expression_type == Expression.INDEX and
        expr.arg[0].expression_type == Expression.VARIABLE):
        var_id = ("index:" +
                  expr.arg[0].variable.variable_id +
                  expression_str.key_str(expr))
        return {var_id: [expr]}

    retval = defaultdict(list)
    for arg in expr.arg:
        for var_id, instances in find_index_var_instances(arg).iteritems():
            retval[var_id] += instances

    return retval

def replace_index_vars(problem):
    index_vars = find_index_var_instances(problem.objective)
    for constr in problem.constraint:
        for var_id, instances in find_index_var_instances(constr).iteritems():
            index_vars[var_id] += instances

    for var_id, instances in index_vars.iteritems():
        old_var = instances[0]
        m, n = old_var.size.dim
        new_var = variable(m, n, var_id)
        problem.constraint.add().CopyFrom(equality_constraint(new_var, old_var))
        for instance in instances:
            instance.CopyFrom(new_var)

    return problem

PROBLEM_TRANSFORMS = [
    replace_index_vars
]

GRAPH_TRANSFORMS = [
    combine_affine_functions,
    move_equality_indicators,
    separate_objective_terms,
]

def transform(problem):
    # for f in PROBLEM_TRANSFORMS:
    #     problem = f(problem)

    graph = ProblemGraph(problem)
    for f in GRAPH_TRANSFORMS:
        f(graph)
    return graph.problem()
