"""Analyze the problem in sum-of-prox form and combine/split terms."""

from collections import defaultdict

from epopt import expression
from epopt import tree_format
from epopt.compiler import validate
from epopt.compiler.problem_graph import *
from epopt.compiler.transforms import linear
from epopt.compiler.transforms.transform_util import *
from epopt.proto.epsilon.expression_pb2 import Expression, ProxFunction
from epopt.util import *

def has_data_constant(expr):
    if (expr.expression_type == Expression.CONSTANT and
        expr.constant.data_location != ""):
        return True

    for arg in expr.arg:
        if has_data_constant(arg):
            return True

    return False

def replace_var(expr, old_var_id, new_var):
    if (expr.expression_type == Expression.VARIABLE and
        expr.variable.variable_id == old_var_id):
        return new_var
    return expression.Expression.FromProto(
        expr.proto,
        [replace_var(arg, old_var_id, new_var) for arg in expr.arg])

def is_prox_friendly_constraint(graph, f):
    """Returns true if f represents a prox-friendly equality constraint.

    In other words, one that can be treated as a constraint without interfering
    with the proximal operators for the other objective terms."""

    if not f.expr.prox_function.prox_function_type == ProxFunction.ZERO:
        return False

    assert len(f.expr.arg) == 1
    for f_var in graph.edges_by_function[f]:
        edges = graph.edges_by_variable[f_var.variable]
        if (len(edges) > 1 and
            not f.expr.arg[0].affine_props.linear_maps[f_var.variable].scalar):
            return False

    return True

def max_overlap_function(graph, f):
    """Return the objective term with maximum overlap in variables."""

    def variables(g):
        return set(g_var.variable for g_var in graph.edges_by_function[g])
    variables_f = variables(f)
    def overlap(g):
        return len(variables(g).intersection(variables_f))

    candidates = [g for g in graph.obj_terms if g != f]
    if not candidates:
        return

    h = max(candidates, key=overlap)

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
        if not f.expr.prox_function.prox_function_type == ProxFunction.AFFINE:
            continue

        # no variables
        if not graph.edges_by_function[f]:
            continue

        # no other functions with overlap
        g = max_overlap_function(graph, f)
        if not g:
            continue

        graph.remove_function(f)
        graph.remove_function(g)

        # Combine functions with non-affine function first
        graph.add_function(
            Function(expression.add(g.expr, f.expr), constraint=False))

def move_equality_indicators(graph):
    """Move certain equality indicators from objective to constraints."""
    # Single prox case, dont move it
    if len(graph.obj_terms) == 1:
        return

    for f in graph.obj_terms:
        if is_prox_friendly_constraint(graph, f):
            # Modify it to be an equality constraint
            f.expr.CopyFrom(expression.indicator(Cone.ZERO, f.expr.arg[0]))
            f.constraint = True


def separate_objective_terms(graph):
    """Add variable copies to make functions separable.

    This applies to objective functions only and we dont need to modify the
    first occurence.
   """
    for var in graph.nodes(VARIABLE):
        functions = graph.neighbors(var, FUNCTION)

        # NOTE(mwytock): We reverse the order of the functions here because we
        # prefer the last function to retain the original variable id
        functions.reverse()

        for f in functions[1:]:
            m, n = dims(var.expr)
            old_var_id = var.expr.variable.variable_id
            new_var_id = "separate:%s:%s" % (old_var_id, f.node_id)

            new_var = graph.add_node(
                expression.variable(m, n, new_var_id), VARIABLE, new_var_id)
            f.expr = replace_var(f.expr, old_var_id, new_var.expr)
            graph.remove_edge(f, var)
            graph.add_edge(f, new_var)

            eq_constr = graph.add_node(linear.transform_expr(
                expression.eq_constraint(new_var.expr, var.expr)), CONSTRAINT)
            graph.add_edge(eq_constr, new_var)
            graph.add_edge(eq_constr, var)

def add_constant_prox(graph):
    """Add f(x) = 0 term for variables only appearing in constraints."""

    for var in graph.nodes(VARIABLE):
        # Only add constant prox for variables not appearing in objective
        if graph.neighbors(var, FUNCTION):
            continue

        f_expr = expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.CONSTANT), var.expr)
        graph.add_edge(graph.add_node(f_expr, FUNCTION), var)

def variables(expr):
    if expr.expression_type == Expression.VARIABLE:
        yield expr
    for arg in expr.arg:
        for var in variables(arg):
            yield var

def add_function(f_expr, node_type, graph):
    var_list = list(variables(f_expr))

    # Exclude constant functions
    if not var_list:
        return

    f = graph.add_node(f_expr, node_type)
    for var_expr in var_list:
        var_id = var_expr.variable.variable_id
        graph.add_edge(f, graph.add_node(var_expr, VARIABLE, node_id=var_id))

def build_graph(problem):
    graph = ProblemGraph()
    for f_expr in problem.objective.arg:
        add_function(f_expr, FUNCTION, graph)
    for constr_expr in problem.constraint:
        add_function(f_expr, CONSTRAINT, graph)
    return graph

# TODO(mwytock): Add back these optimizations when ready
# combine_affine_functions
# move_equality_indicators,

GRAPH_TRANSFORMS = [
    separate_objective_terms,
    add_constant_prox,
]

def transform_problem(problem):
    validate.check_sum_of_prox(problem)
    graph = build_graph(problem)
    if not graph.nodes(VARIABLE):
        return problem

    for f in GRAPH_TRANSFORMS:
        f(graph)
        log_debug(
            lambda f, graph:
            "%s:\n%s" %
            (f.__name__,
             tree_format.format_problem(graph.problem)),
            f, graph)
    return graph.problem
