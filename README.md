# Epsilon [![Circle CI](https://circleci.com/gh/mwytock/epsilon.svg?style=svg)](https://circleci.com/gh/mwytock/epsilon)

Epsilon is a general convex solver based on functions with efficient proximal
operators. See [Wytock et al., Convex programming with fast proximal and linear
operators](http://arxiv.org/abs/1511.04815) for technical details.

## Installation instructions

### NumPy, SciPy and CVXPY dependencies

Epsilon requires the basic numerical python environment that is also required
for CVXPY. More [[detailed instructions][http://www.cvxpy.org/en/latest/install/index.html]] are available from the CVXPY
website, in essence you need to first install the latest version of NumPy/SciPy:

```
pip install -U numpy scipy
pip install -U cvxpy
```

### Installation with pip

Once CVXPY is installed, Epsilon can be installed with pip

#### Mac OS X

```
pip install epopt
```

#### Linux

```
pip install http://epopt.s3.amazonaws.com/epopt-0.1.0-cp27-none-linux_x86_64.whl
```

## Usage

In order to use Epsilon, form an optimization problem using CVXPY in the usual
way but solve it using Epsilon.
```python
import numpy as np
import cvxpy as cp
import epopt as ep

# Form lasso problem with CVXPY
m = 5
n = 10
A = np.random.randn(m,n)
b = np.random.randn(m)
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A*x - b) + cp.norm1(x)))

# Solve with Epsilon
print "Objective:"
print ep.solve(prob)
print "Solution:"
print x.value
```

## Development instructions

These instructions are for setting up the development environment required to
compile Epsilon from source. The package is also pip-installable, and is
intended to be used in this fashion by end-users.

### Build third-party C++ dependencies

First, compile the third-party C++ dependencies (gflags, glog, protobuf) which
are provided as git submodules linked from the main repository
```
git submodule update --init
```

A script is provided to to compile these libraries which must be built before
compiling Epsilon.
```
tools/build_third_party.sh
```

### Build Epsilon and run tests

Compile the C++ code and run tests
```
make -j test
```

Now build the C++ Python extension and set up the local development environment
```
python setup.py build
python setup.py develop --user
```
Run python tests
```
pip install nose
nosetests epopt
```

## Benchmark results

### Epsilon
```
python -m epopt.problems.benchmark
```
Problem       |   Time | Objective
:------------- | ------:| ---------:
basis_pursuit  |   2.96s|   1.45e+02
covsel         |   0.77s|   3.63e+02
group_lasso    |   9.61s|   1.61e+02
hinge_l1       |   5.42s|   1.50e+03
huber          |   0.51s|   2.18e+03
lasso          |   3.88s|   1.64e+01
least_abs_dev  |   0.38s|   7.09e+03
logreg_l1      |   4.84s|   1.04e+03
lp             |   0.56s|   7.77e+02
mnist          |   1.44s|   1.53e+03
quantile       |  16.40s|   3.64e+03
tv_1d          |   0.50s|   2.13e+05
tv_denoise     |  24.17s|   1.15e+06

### SCS
```
python -m epopt.problems.benchmark --scs
```

 Problem       |   Time | Objective
:------------- | ------:| ---------:
basis_pursuit  |  16.99s|   1.45e+02
covsel         |  23.50s|   3.62e+02
group_lasso    |  23.31s|   1.61e+02
hinge_l1       |  52.62s|   1.50e+03
huber          |   3.39s|   2.18e+03
lasso          |  22.02s|   1.63e+01
least_abs_dev  |   3.81s|   7.10e+03
logreg_l1      |  55.53s|   1.04e+03
lp             |   5.47s|   7.75e+02
mnist          | 227.65s|   1.60e+03
tv_1d          |  47.28s|   3.51e+05
tv_denoise     | 372.86s|   1.69e+06
quantile       |  88.60s|   4.99e+03
