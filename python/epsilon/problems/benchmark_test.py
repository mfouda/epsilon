
from epsilon.problems import benchmark
from epsilon.problems import lasso
from epsilon.problems.problem_instance import ProblemInstance

def test_benchmarks():
    benchmark.print_benchmarks(
        [ProblemInstance("lasso", lasso.create, dict(m=5, n=10))])
