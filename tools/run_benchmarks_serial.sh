#!/bin/bash -u
#
# Run Epsilon benchmarks using GNU parallel. 

benchmark="python -m epsilon.problems.benchmark"

if [ $# -eq 1 ]; then
    names=$1
else
    names=$($benchmark --list-benchmarks)
fi
problems=$($benchmark --list-problems)
time=3600

# Epsilon is not multi-threaded (yet) but BLAS implementations used as a
# backend to numpy can be.
export OPENBLAS_NUM_THREADS=1

for problem in $problems; do
    for name in $names; do
	timelimit -t$time $benchmark --benchmark $name --problem $problem
    done
done
