#!/bin/bash -eu
#
# Run Epsilon benchmarks using GNU parallel. This seems to offer
# better isolation than Python's multiprocessing framework.

benchmark="python -m epsilon.problems.benchmark"

# Epsilon is not multi-threaded (yet) but BLAS implementations used as a
# backend to numpy can be.
export OPENBLAS_NUM_THREADS=1

parallel python -m epsilon.problems.benchmark \
	 --no-header \
	 $* \
	 --problem ::: \
	 $($benchmark --list)

