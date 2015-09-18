import logging

from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import finalize
from epsilon.compiler import linearize
from epsilon.compiler import recombine
from epsilon.compiler import separate
from epsilon.expression_str import problem_str


TRANSFORMS = [
    attributes.transform,
    canonicalize.transform,
    linearize.transform,
    recombine.transform,
    separate.transform,
    finalize.transform,
]

def compile(problem):
    logging.debug("Compiler input:\n%s", problem_str(problem))
    for transform in TRANSFORMS:
        problem = transform(problem)
        logging.debug("Intermediate:\n%s", problem_str(problem))
    logging.debug("Compiler output:\n%s", problem_str(problem))
    return problem
