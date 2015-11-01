#!/bin/bash -eu
#
# Run Epsilon benchmarks using GNU parallel. This seems to offer
# better isolation than Python's multiprocessing framework.

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

parallel timelimit -t$time $benchmark \
	 --benchmark {1} \
	 --problem {2} \
	 ::: $names ::: $problems
