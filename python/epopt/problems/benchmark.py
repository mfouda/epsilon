#!/usr/bin/env python

import argparse
import logging
import sys

import cvxpy as cp
import numpy as np

from epopt import cvxpy_expr
from epopt import cvxpy_solver
from epopt.compiler import compiler
from epopt.problems import *
from epopt.problems import benchmark_util

from epopt.problems.problem_instance import ProblemInstance

# Need to fix ATA, maybe faster sparse matrix ops?
# ProblemInstance("portfolio", portfolio.create, dict(m=500, n=500000)),

# Slow, maybe consider a smaller version?
# ProblemInstance("mv_lasso_sparse", lasso.create, dict(m=1500, n=50000, k=10, rho=0.01, mu=0.1)),

# Fix general A for least squares
# ProblemInstance("group_lasso", group_lasso.create, dict(m=1500, ni=50, K=200)),
# ProblemInstance("quantile", quantile.create, dict(m=400, n=10, k=50, p=1)),

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=1000, n=3000)),
    ProblemInstance("covsel", covsel.create, dict(m=100, n=200, lam=0.1)),
    ProblemInstance("fused_lasso", fused_lasso.create, dict(m=1000, ni=10, k=1000)),
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
    ProblemInstance("robust_svm", robust_svm.create, dict(m=5000, n=1500)),
    ProblemInstance("tv_1d", tv_1d.create, dict(n=100000)),
]

PROBLEMS_SCALE = []
PROBLEMS_SCALE += [ProblemInstance(
    "lasso_%d" % int(m),
    lasso.create,
    dict(m=int(m), n=10*int(m), rho=1 if m < 50 else 0.01))
    for m in np.logspace(1, np.log10(5000), 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "mv_lasso_%d" % int(m),
    lasso.create,
    dict(m=int(m), n=10*int(m), k=10, rho=1 if m < 50 else 0.01))
    for m in np.logspace(1, np.log10(5000), 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "fused_lasso_%d" % int(m),
    fused_lasso.create,
    dict(m=int(m), ni=10, k=int(m)))
    for m in np.logspace(1, 3, 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "hinge_l2_%d" % int(n),
    hinge_l2.create,
    dict(m=10*int(n), n=int(n)))
    for n in np.logspace(1, np.log10(5000), 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "robust_svm_%d" % int(n),
    robust_svm.create,
    dict(m=3*int(n), n=int(n)))
    for n in np.logspace(1, np.log10(1500), 20)]

def benchmark_epsilon(cvxpy_prob):
    cvxpy_solver.solve(cvxpy_prob, rel_tol=1e-2, abs_tol=1e-4)
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
    parser.add_argument("--benchmark", default="epsilon")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--list-problems", action="store_true")
    parser.add_argument("--problem")
    parser.add_argument("--problem-match")
    parser.add_argument("--problem-set", default="PROBLEMS")
    parser.add_argument("--scs-indirect", action="store_true")
    parser.add_argument("--write")
    args = parser.parse_args()

    problems = locals()[args.problem_set]
    if args.problem:
        problems = [p for p in problems if p.name == args.problem]
    elif args.problem_match:
        problems = [
            p for p in problems if p.name.startswith(args.problem_match)]

    if args.list_problems:
        for problem in problems:
            print problem.name
        sys.exit(0)

    if args.list_benchmarks:
        for benchmark in BENCHMARKS:
            print benchmark
        sys.exit(0)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.write:
        benchmark_util.write_problems(problems, args.write)
        sys.exit(0)

    for result in run_benchmarks([args.benchmark], problems):
        print "\t".join(str(x) for x in result)

else:
    args = argparse.Namespace()
