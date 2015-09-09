# Epsilon
Epsilon is a general convex solver based on functions with efficient proximal
operators.

Installation
============
The epsilon C++ code currently has some standard library dependencies which are
not bundled as part of the python package. These must be installed before
compiling and installing the epsilon package with `pip`.

Mac OS X
--------
Using homebrew

```
brew install glog
brew install --devel protobuf
pip install epsilon
```

Ubuntu
------
First download and install the protocol buffer library (must be >3.0.0 which is
not yet included in apt-get) from https://github.com/google/protobuf.

Then,
```
apt-get install libglog-dev
pip install epsilon
````