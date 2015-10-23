import logging

from epsilon.compiler import canonicalize
from epsilon.compiler import canonicalize_linear
from epsilon.compiler import combine
from epsilon import tree_format

TRANSFORMS = [
    canonicalize.transform,
    combine.transform,
    canonicalize_linear.transform_problem,
]

def compile_problem(problem):
    logging.debug("Compiler input:\n%s", tree_format.format_problem(problem))
    for transform in TRANSFORMS:
        problem = transform(problem)
        logging.debug("Intermediate:\n%s", tree_format.format_problem(problem))
    return problem
