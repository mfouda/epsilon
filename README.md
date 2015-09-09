# Epsilon [![Circle CI](https://circleci.com/gh/mwytock/epsilon.svg?style=svg)](https://circleci.com/gh/mwytock/epsilon)

Epsilon is a general convex solver based on functions with efficient proximal
operators.

## Installation

The epsilon C++ code has library dependencies which are not bundled as part of
the python package. These must be installed before the `epsilon` package itself.

We assume that CVXPY has already been installed, see instructions at
http://www.cvxpy.org/en/latest/install/index.html.

### Dependencies on Mac OS X

Install C++ dependencies using Homebrew (or MacPorts):

```
brew install glog gflags
brew install --devel protobuf
```
Install epsilon and run tests with nose
```
pip install epsilon
nosetests epsilon
```

### Dependencies on Ubuntu

First download and install the protocol buffer library (must be >3.0.0 which is
not yet included in apt-get) from https://github.com/google/protobuf.
```
wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-1/protobuf-cpp-3.0.0-beta-1.tar.gz
tar zxvf protobuf-cpp-3.0.0-beta-1.tar.gz
cd protobuf-cpp-3.0.0-beta-1
./configure
make install
```
The other dependencies can be installed using the distribution package manager
```
apt-get install libglog-dev libgflags-dev
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
