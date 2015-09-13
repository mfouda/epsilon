from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import split

TRANSFORMS = [
    attributes.transform,
    canonicalize.transform,
    split.transform
]

def compile(problem):
    for transform in TRANSFORMS:
        problem = transform(problem)
    return problem
