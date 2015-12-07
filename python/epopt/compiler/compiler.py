import logging

from epopt.compiler.transforms import prox
from epopt.compiler.transforms import separate
from epopt import tree_format

TRANSFORMS = [
    prox.transform_problem,
    separate.transform_problem,
]

def transform_name(transform):
    return ".".join((transform.__module__, transform.__name__))

def compile_problem(problem):
    logging.debug("input:\n%s", tree_format.format_problem(problem))
    for transform in TRANSFORMS:
        problem = transform(problem)
        logging.debug(
            "%s:\n%s",
            transform_name(transform),
            tree_format.format_problem(problem))
    return problem
