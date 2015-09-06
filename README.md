# Epsilon
Epsilon is a general convex solver based on functions with efficient proximal
operators.

Compiling on OS X
-----------------

We need a few C++ library dependencies which are available through various
package managers, including Homebrew.
```
brew install gflags glog gperftools parallel
```
We also need the latest development version of the protobuf library (v3.0.0 or
higher)
```
brew install --devel protobuf
```
Compile and run tets
```
make test
```

Compiling on Ubuntu
-------------------

Again, install the necessary dependencies
```
apt-get install libgflags-dev libglog-dev gperftools-dev parallel
```
and download and compile the latest development version of protobuf from
https://github.com/google/protobuf

Compile and run tets
```
make test
```
