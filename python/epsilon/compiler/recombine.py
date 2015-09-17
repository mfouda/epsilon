"""Split/merge proximal operators to improve efficiency."""

from collections import defaultdict

from epsilon.compiler import compiler_error
from epsilon.expression_pb2 import Expression, Curvature, Problem
from epsilon.expression import *

def merge_affine(problem):
    """Merge affine terms with other proximal operators."""

    non_affine = []
    affine = []
    for f in problem.objective.arg:
        f_vars = set(expr_vars(f).keys())
        if f.curvature.curvature_type == Curvature.AFFINE:
            affine.append((f, f_vars))
        else:
            non_affine.append((f, f_vars))

    # All objective functions are affine, nothing to do
    if not non_affine:
        return problem

    final = []
    for f, f_vars in affine:
        g, g_vars = max(
            non_affine, key=lambda (g, g_vars): len(f_vars.intersection(g_vars)))
        g.CopyFrom(add(g, f))

    return Problem(
        objective=add(*(f for f, _ in non_affine)),
        constraint=problem.constraint)


def transform(problem):
    if problem.objective.expression_type != Expression.ADD:
        raise RecombineError("Objective root expression is not add", problem)

    return merge_affine(problem)
