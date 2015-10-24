#!/usr/bin/env python

import argparse
import logging
import time
import sys

import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.compiler import compiler
from epsilon.problems import *
from epsilon.problems import benchmark_format
from epsilon.problems import benchmark_util

from epsilon.problems.problem_instance import ProblemInstance
from epsilon.problems.benchmark_format import Column

# Add back when we have convolution
# ProblemInstance("tv_denoise", tv_denoise.create, dict(n=400, lam=1)),

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=1000, n=3000)),
    ProblemInstance("covsel", covsel.create, dict(m=100, n=200, lam=0.1)),
    ProblemInstance("group_lasso", group_lasso.create, dict(m=1500, ni=50, K=200)),
    ProblemInstance("hinge_l1", hinge_l1.create, dict(m=1500, n=5000)),
    ProblemInstance("huber", huber.create, dict(m=5000, n=200)),
    ProblemInstance("lasso", lasso.create, dict(m=1500, n=5000)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=5000, n=200)),
    ProblemInstance("logreg_l1", logreg_l1.create, dict(m=1500, n=5000)),
    ProblemInstance("lp", lp.create, dict(m=800, n=1000)),
    ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_SMALL, n=1000)),
    ProblemInstance("quantile", quantile.create, dict(m=400, n=20, k=100)),
    ProblemInstance("tv_1d", tv_1d.create, dict(n=100000)),
]

FORMATTERS = {
    "text": benchmark_format.Text,
    "html": benchmark_format.HTML,
    "latex": benchmark_format.Latex,
}

def cvxpy_kwargs(solver):
    return kwargs

def benchmark_epsilon(cvxpy_prob):
    params = solver_params_pb2.SolverParams(rel_tol=1e-3)
    solve.solve(cvxpy_prob, params=params)
    return cvxpy_prob.objective.value

def benchmark_cvxpy(solver, cvxpy_prob):
    kwargs = {"solver": solver}
    if solver == cp.SCS:
        kwargs["use_indirect"] = args.scs_indirect
        kwargs["max_iters"] = 10000

    try:
        # TODO(mwytock): Probably need to run this in a separate thread/process
        # and kill after one hour?
        cvxpy_prob.solve(**kwargs)
    except cp.error.SolverError:
        # Raised when solver cant handle a problem
        return float("nan")

    return cvxpy_prob.objective.value

def benchmark_cvxpy_canon(solver, cvxpy_prob):
    cvxpy_prob.get_problem_data(solver=solver)

def run_benchmarks(benchmarks, problems):
    for problem in problems:
        logging.debug("problem %s", problem.name)
        cvxpy_prob = problem.create()
        data = [problem.name]
        for benchmark in benchmarks:
            logging.debug("running %s", benchmark)
            t0 = time.time()
            result = benchmark(cvxpy_prob)
            t1 = time.time()
            data.append(t1 - t0)
            logging.debug("done %f seconds", t1-t0)
            if result:
                data.append(result)
        yield data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--problem")
    parser.add_argument("--scs-indirect", action="store_true")
    parser.add_argument("--format", default="text")
    parser.add_argument("--include-cvxpy", action="store_true")
    parser.add_argument("--write")

    args = parser.parse_args()

    if args.problem:
        problems = [p for p in PROBLEMS if p.name == args.problem]
    else:
        problems = PROBLEMS

    if args.write:
        benchmark_util.write_problems(problems, args.write)
        sys.exit(0)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Epsilon benchmarks
    benchmarks = [benchmark_epsilon]
    super_columns = []
    columns = [
        Column("Problem",   15, "%-15s"),
        # Epsilon
        Column("Time",      8,  "%7.2fs", right=True),
        Column("Objective", 11, "%11.2e", right=True),
    ]

    # Optionally add CVXPY benchmarks
    if args.include_cvxpy:
        super_columns = [
            Column("",           15),
            Column("Epsilon",    20, right=True, colspan=2),
            #Column("CVXPY",      8, right=True),
            Column("CVXPY+SCS",  20, right=True, colspan=2),
            Column("CVXPY+ECOS", 20, right=True, colspan=2),
        ]

        columns += [
            # CVXPY canon
            #Column("Time",      8,  "%7.2fs", right=True),
            # CVXPY + SCS
            Column("Time",      8,  "%7.2fs", right=True),
            Column("Objective", 11, "%11.2e", right=True),
            # CVXPY + ECOS
            Column("Time",      8,  "%7.2fs", right=True),
            Column("Objective", 11, "%11.2e", right=True),
        ]

        benchmarks += [
            # NOTE(mwytock): We assume that get_problem_data() times are
            # similar for ECOS and SCS so we only include one
            #lambda p: benchmark_cvxpy_canon(cp.SCS, p),
            lambda p: benchmark_cvxpy(cp.SCS, p),
            lambda p: benchmark_cvxpy(cp.ECOS, p)
        ]

    formatter = FORMATTERS[args.format](super_columns, columns)
    formatter.print_header()
    for row in run_benchmarks(benchmarks, problems):
        formatter.print_row(row)
    formatter.print_footer()

else:
    args = argparse.Namespace()
