import logging

from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import combine
from epsilon.expression_str import problem_str

TRANSFORMS = [
    canonicalize.transform,
    combine.transform,
]

def compile(problem):
    logging.debug("Compiler input:\n%s", problem_str(problem))
    for transform in TRANSFORMS:
        problem = transform(problem)
        logging.debug("Intermediate:\n%s", problem_str(problem))
    return problem
