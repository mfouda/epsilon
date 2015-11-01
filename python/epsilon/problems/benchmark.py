#!/usr/bin/env python

import argparse
import logging
import sys

import cvxpy as cp
import numpy as np

from epsilon import cvxpy_expr
from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.compiler import compiler
from epsilon.problems import *
from epsilon.problems import benchmark_util

from epsilon.problems.problem_instance import ProblemInstance

# Need to fix ATA, maybe faster sparse matrix ops?
# ProblemInstance("portfolio", portfolio.create, dict(m=500, n=500000)),

# Slow, maybe consider a smaller version?
# ProblemInstance("mv_lasso_sparse", lasso.create, dict(m=1500, n=50000, k=10, rho=0.01, mu=0.1)),

# Verify choice of lambda

# Need to imporve convergence
# ProblemInstance("quantile", quantile.create, dict(m=400, n=5, k=100)),

# Fix general A for least squares
# ProblemInstance("group_lasso", group_lasso.create, dict(m=1500, ni=50, K=200)),


PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=1000, n=3000)),
    ProblemInstance("covsel", covsel.create, dict(m=100, n=200, lam=0.1)),
    ProblemInstance("hinge_l1", hinge_l1.create, dict(m=1500, n=5000, rho=0.01)),
    ProblemInstance("hinge_l1_sparse", hinge_l1.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
    ProblemInstance("hinge_l2", hinge_l2.create, dict(m=5000, n=1500)),
    ProblemInstance("hinge_l2_sparse", hinge_l2.create, dict(m=10000, n=1500, mu=0.1)),
    ProblemInstance("huber", huber.create, dict(m=5000, n=200)),
    ProblemInstance("lasso", lasso.create, dict(m=1500, n=5000, rho=0.01)),
    ProblemInstance("lasso_sparse", lasso.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=5000, n=200)),
    ProblemInstance("logreg_l1", logreg_l1.create, dict(m=1500, n=5000, rho=0.01)),
    ProblemInstance("logreg_l1_sparse", logreg_l1.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
    ProblemInstance("lp", lp.create, dict(m=800, n=1000)),
    ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_SMALL, n=1000)),
    ProblemInstance("mv_lasso", lasso.create, dict(m=1500, n=5000, k=10, rho=0.01)),
    ProblemInstance("qp", qp.create, dict(n=1000)),
    ProblemInstance("robust_pca", robust_pca.create, dict(n=100)),
    ProblemInstance("tv_1d", tv_1d.create, dict(n=100000)),
]

def benchmark_epsilon(cvxpy_prob):
    params = solver_params_pb2.SolverParams(rel_tol=1e-3, abs_tol=1e-5)
    solve.solve(cvxpy_prob, params=params)
    return cvxpy_prob.objective.value

def benchmark_cvxpy(solver, cvxpy_prob):
    kwargs = {"solver": solver,
              "verbose": args.debug}
    if solver == cp.SCS:
        kwargs["use_indirect"] = args.scs_indirect
        kwargs["max_iters"] = 10000

    try:
        # TODO(mwytock): ProblemInstanceably need to run this in a separate thread/process
        # and kill after one hour?
        cvxpy_prob.solve(**kwargs)
        return cvxpy_prob.objective.value
    except cp.error.SolverError:
        # Raised when solver cant handle a problem
        return float("nan")

BENCHMARKS = {
    "epsilon": benchmark_epsilon,
    "scs": lambda p: benchmark_cvxpy(cp.SCS, p),
    "ecos": lambda p: benchmark_cvxpy(cp.ECOS, p),
}

def benchmark_cvxpy_canon(solver, cvxpy_prob):
    cvxpy_prob.get_problem_data(solver=solver)

def run_benchmarks(benchmarks, problems):
    for problem in problems:
        logging.debug("problem %s", problem.name)

        t0 = benchmark_util.cpu_time()
        np.random.seed(0)
        cvxpy_prob = problem.create()
        t1 = benchmark_util.cpu_time()
        logging.debug("creation time %f seconds", t1-t0)

        data = [problem.name]
        for benchmark in benchmarks:
            logging.debug("running %s", benchmark)

            t0 = benchmark_util.cpu_time()
            value = BENCHMARKS[benchmark](cvxpy_prob)
            t1 = benchmark_util.cpu_time()

            logging.debug("done %f seconds", t1-t0)
            yield benchmark, "%-15s" % problem.name, t1-t0, value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--problem")
    parser.add_argument("--benchmark", default="epsilon")
    parser.add_argument("--scs-indirect", action="store_true")
    parser.add_argument("--write")
    parser.add_argument("--list-problems", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    args = parser.parse_args()

    if args.list_problems:
        for problem in PROBLEMS:
            print problem.name
        sys.exit(0)

    if args.list_benchmarks:
        for benchmark in BENCHMARKS:
            print benchmark
        sys.exit(0)

    if args.problem:
        problems = [p for p in PROBLEMS if p.name == args.problem]
    else:
        problems = PROBLEMS

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.write:
        benchmark_util.write_problems(problems, args.write)
        sys.exit(0)

    for result in run_benchmarks([args.benchmark], problems):
        print "\t".join(str(x) for x in result)

else:
    args = argparse.Namespace()
