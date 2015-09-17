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

  Problem |   Time | Objective
:-------- | ------:| ---------:
covsel    |   0.38s|   3.68e+02
lasso     |   3.83s|   1.63e+01
tv_smooth |  17.32s|   1.15e+06

### SCS
```
python -m epsilon.problems.benchmark --scs
```

  Problem |   Time | Objective
:-------- | ------:| ---------:
covsel    |  25.56s|   3.71e+02
lasso     |  20.04s|   1.63e+01
tv_smooth | 402.10s|   1.80e+06