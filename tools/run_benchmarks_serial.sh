#!/bin/bash -u

cmd="python -m epsilon.problems.benchmark $*"

benchmarks=$($cmd --list-benchmarks)
problems=$($cmd --list-problems)
time=3600

# Epsilon is not multi-threaded (yet) but BLAS implementations used as a
# backend to numpy can be.
export OPENBLAS_NUM_THREADS=1

for problem in $problems; do
    for benchmark in $benchmarks; do
	timelimit -t$time $cmd --benchmark $benchmark --problem $problem
    done
done
