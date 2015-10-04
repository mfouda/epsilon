#!/usr/bin/env python

from collections import namedtuple
import argparse
import errno
import logging
import os
import time

import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon import solve
from epsilon import solver_params_pb2
from epsilon.compiler import compiler
from epsilon.expression_pb2 import Expression
from epsilon.problems import basis_pursuit
from epsilon.problems import covsel
from epsilon.problems import group_lasso
from epsilon.problems import hinge_l1
from epsilon.problems import huber
from epsilon.problems import lasso
from epsilon.problems import least_abs_dev
from epsilon.problems import logreg_l1
from epsilon.problems import lp
from epsilon.problems import mnist
from epsilon.problems import quantile
from epsilon.problems import tv_1d
from epsilon.problems import tv_denoise
from epsilon.problems.problem_instance import ProblemInstance

class Column(namedtuple("Column", ["name", "width", "fmt", "right"])):
    """Columns for a Markdown appropriate text table."""

    @property
    def header(self):
        align = "" if self.right else "-"
        header_fmt = " %" + align + str(self.width-2) + "s "
        return header_fmt % self.name

    @property
    def sub_header(self):
        val = "-" * (self.width-2)
        if self.right:
            val = " " + val + ":"
        else:
            val = ":" + val + " "
        return val

Column.__new__.__defaults__ = (None, None, None, False)

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
    ProblemInstance("tv_1d", tv_1d.create, dict(n=100000)),
    ProblemInstance("tv_denoise", tv_denoise.create, dict(n=400, lam=1)),
]

COLUMNS = [
    Column("Problem",   15, "%-15s"),
    Column("Time",      8,  "%7.2fs", right=True),
    Column("Objective", 11, "%11.2e", right=True),
]

def print_header():
    print "|".join(c.header for c in COLUMNS)
    print "|".join(c.sub_header for c in COLUMNS)

def print_result(*args):
    print "|".join(c.fmt % args[i] for i, c in enumerate(COLUMNS))

def run_benchmarks(problems):
    for problem in problems:
        if args.problem and problem.name != args.problem:
            continue

        cvxpy_prob = problem.create()

        t0 = time.time()
        if args.scs:
            cvxpy_prob.solve(
                solver=cp.SCS, verbose=args.debug,
                use_indirect=args.scs_indirect)
        else:
            params = solver_params_pb2.SolverParams(rel_tol=1e-3)
            solve.solve(cvxpy_prob, params=params)
        t1 = time.time()

        yield problem.name, t1-t0, cvxpy_prob.objective.value

def print_benchmarks(problems):
    print_header()
    for result in run_benchmarks(problems):
        print_result(*result)

def modify_data_location(expr, f):
    if (expr.expression_type == Expression.CONSTANT and
        expr.constant.data_location != ""):
        expr.constant.data_location = f(expr.constant.data_location)

    for arg in expr.arg:
        modify_data_location(arg, f)

def makedirs_existok(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def write_benchmarks(problems, location):
    mem_prefix = "/mem/"
    file_prefix = "/local" + location + "/"
    def rewrite_location(name):
        assert name[:len(mem_prefix)] == mem_prefix
        return file_prefix + name[len(mem_prefix):]

    makedirs_existok(location)
    for problem in problems:
        prob_proto, data_map = cvxpy_expr.convert_problem(problem.create())
        prob_proto = compiler.compile(prob_proto)

        modify_data_location(prob_proto.objective, rewrite_location)
        for constraint in prob_proto.constraint:
            modify_data_location(constraint, rewrite_location)

        with open(os.path.join(location, problem.name), "w") as f:
            f.write(prob_proto.SerializeToString())

        for name, value in data_map.items():
            assert name[:len(mem_prefix)] == mem_prefix
            filename = os.path.join(location, name[len(mem_prefix):])
            makedirs_existok(os.path.dirname(filename))
            with open(filename, "w") as f:
                f.write(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scs", action="store_true")
    parser.add_argument("--scs-indirect", action="store_true")
    parser.add_argument("--write")
    parser.add_argument("--problem")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.write:
        write_benchmarks(PROBLEMS, args.write)
    else:
        print_benchmarks(PROBLEMS)
else:
    args = argparse.Namespace()
    args.scs = False
    args.problem = ""
