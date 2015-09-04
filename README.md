# Epsilon
Epsilon is a general convex solver based on functions with efficient proximal
operators.

Compiling on OS X
-----------------

We need a few C++ library dependencies which are available through various
package managers, including Homebrew.
```
brew install gflags glog gperftools
```
We also need the latest development version of the protobuf library (v3.0.0 or
higher)
```
brew install --devel protobuf
```

Compiling on Ubuntu
-------------------
