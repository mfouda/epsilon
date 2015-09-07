# Epsilon
Epsilon is a general convex solver based on functions with efficient proximal
operators.

Compiling on OS X
-----------------

We also need the latest development version of the protobuf library (v3.0.0 or
higher)
```
brew install --devel protobuf
```
Build and install
```
python python/setup.py install
```

Compiling on Ubuntu
-------------------

Download and compile the latest development version of protobuf from
https://github.com/google/protobuf

Compile and run tets
```
python python/setup.py install
```
