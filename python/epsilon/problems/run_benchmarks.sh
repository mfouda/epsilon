#!/bin/bash -eu
#
# Run Epsilon benchmarks using GNU parallel. This seems to offer
# better isolation than Python's multiprocessing framework.

benchmark="python -m epsilon.problems.benchmark"

# Epsilon is not multi-threaded (yet) but numpy BLAS implementations
# can be.  
export OPENBLAS_NUM_THREADS=1

parallel python -m epsilon.problems.benchmark \
	 --no-header \
	 $* \
	 --problem ::: \
	 $($benchmark --list)

