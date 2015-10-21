#!/usr/bin/env python
#
# Usage:
#   python -m epsilon.problems.print lasso '{"m":10 "n":5}'

import argparse
import json

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon import text_format
from epsilon.compiler import canonicalize
from epsilon.compiler import combine
from epsilon.problems import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("kwargs", help="Problem arg, e.g. {\"m\": 10}")
    args = parser.parse_args()

    cvxpy_prob = locals()[args.problem].create(**json.loads(args.kwargs))
    problem = cvxpy_expr.convert_problem(cvxpy_prob)[0]

    print "Original:"
    print text_format.format(problem)

    problem = canonicalize.transform(problem)
    print
    print "Canonicalization:"
    print text_format.format(problem)

    problem = combine.transform(problem)
    print
    print "Separation:"
    print text_format.format(problem)
