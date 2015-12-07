
import argparse

from epsilon.problems import benchmark
from epsilon.problems import lasso
from epsilon.problems.problem_instance import ProblemInstance


def test_benchmarks():
    benchmark.run_benchmarks(
        [lambda p: 0],
        [ProblemInstance("lasso", lasso.create, dict(m=5, n=10))])
