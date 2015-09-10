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

```sh
brew install glog gflags
brew install --devel protobuf
```

### Dependencies on Ubuntu

Install dependencies with the package manager
```sh
apt-get install libglog-dev libgflags-dev
```

The protocol buffer library must be >3.0.0 which is not yet included in
apt-get. It can be downloaed from https://github.com/google/protobuf.
```sh
wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-1/protobuf-cpp-3.0.0-beta-1.tar.gz
tar zxvf protobuf-cpp-3.0.0-beta-1.tar.gz
cd protobuf-cpp-3.0.0-beta-1
./configure
make install
```

### Install Epsilon

Install epsilon
```sh
pip install epsilon
```
and (optionally) run tests with nose
```sh
pip install nose
nosetests epsilon
```

## Benchmark results

Benchmark of epsilon on a suite of common problems
```sh
python -m epsilon.benchmark
```

  Problem |   Time | Objective
:-------- | ------:| ---------:
covsel    |   0.38s|   3.68e+02
lasso     |   3.83s|   1.63e+01
tv_smooth |  17.32s|   1.15e+06
