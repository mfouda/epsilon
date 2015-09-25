# Epsilon [![Circle CI](https://circleci.com/gh/mwytock/epsilon.svg?style=svg)](https://circleci.com/gh/mwytock/epsilon)

Epsilon is a general convex solver based on functions with efficient proximal
operators.

## Installation

The epsilon C++ code has library dependencies which are not bundled as part of
the python package. These must be installed before the epsilon package itself.

We assume that CVXPY has already been installed, see instructions at
http://www.cvxpy.org/en/latest/install/index.html.

### Dependencies on Mac OS X

Install dependencies using Homebrew (or MacPorts):

```
brew install glog gflags
brew install --devel protobuf
```

### Dependencies on Ubuntu

Install dependencies with the package manager
```
apt-get install libglog-dev libgflags-dev
```

The protocol buffer library must be >3.0.0 which is not yet included in
apt-get. It can be downloaed from https://github.com/google/protobuf.
```
wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-1/protobuf-cpp-3.0.0-beta-1.tar.gz
tar zxvf protobuf-cpp-3.0.0-beta-1.tar.gz
cd protobuf-cpp-3.0.0-beta-1
./configure
make install
```

### Install Epsilon

Install epsilon
```
pip install epsilon
```
and (optionally) run tests with nose
```
pip install nose
nosetests epsilon
```

## Benchmark results

### Epsilon
```
python -m epsilon.problems.benchmark
```
 Problem       |   Time | Objective
:------------- | ------:| ---------:
basis_pursuit  |   1.47s|   1.44e+02
covsel         |   0.46s|   3.63e+02
group_lasso    |  10.33s|   1.66e+02
huber          |   0.49s|   2.18e+03
lasso          |   3.93s|   1.71e+01
least_abs_dev  |   0.39s|   7.10e+03
lp             |   0.33s|   7.77e+02
tv_1d          |  13.96s|   1.86e+04
tv_denoise     |  19.56s|   1.15e+06

### SCS
```
python -m epsilon.problems.benchmark --scs
```

 Problem       |   Time | Objective
:------------- | ------:| ---------:
basis_pursuit  |  16.99s|   1.45e+02
covsel         |  23.50s|   3.62e+02
group_lasso    |  23.31s|   1.61e+02
huber          |   3.39s|   2.18e+03
lasso          |  22.02s|   1.63e+01
lp             |   5.47s|   7.75e+02
least_abs_dev  |   3.81s|   7.10e+03
tv_1d          |   0.61s|   2.95e+04
tv_denoise      | 372.86s|   1.69e+06
