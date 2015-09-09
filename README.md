# Epsilon [![Circle CI](https://circleci.com/gh/mwytock/epsilon.svg?style=svg)](https://circleci.com/gh/mwytock/epsilon)

Epsilon is a general convex solver based on functions with efficient proximal
operators.

## Installation

We assume that CVXPY has already been installed, see instructions at
http://www.cvxpy.org/en/latest/install/index.html.

The epsilon C++ code has library dependencies which are not bundled as part of
the python package. These must be installed before the `epsilon` package itself.

### Installation on Mac OS X

Using homebrew

```
brew install glog gflags
brew install --devel protobuf
pip install epsilon
```
Run tests with nose
```
nosetests epsilon
```

### Installation on Ubuntu

First download and install the protocol buffer library (must be >3.0.0 which is
not yet included in apt-get) from https://github.com/google/protobuf.
```
wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-1/protobuf-cpp-3.0.0-beta-1.tar.gz
tar zxvf protobuf-cpp-3.0.0-beta-1.tar.gz
cd protobuf-cpp-3.0.0-beta-1
./configure
make install
```
Then, continue by using `apt-get`
```
apt-get install libglog-dev libgflags-dev
pip install epsilon
```
Run tests with nose
```
nosetests epsilon
```