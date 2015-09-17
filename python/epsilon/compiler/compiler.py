from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import linearize
from epsilon.compiler import recombine
from epsilon.compiler import separate

TRANSFORMS = [
    attributes.transform,
    canonicalize.transform,
    linearize.transform,
    recombine.transform,
    separate.transform
]

def compile(problem):
    for transform in TRANSFORMS:
        problem = transform(problem)
    return problem
