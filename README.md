# Epsilon
Epsilon is a general convex solver based on functions with efficient proximal
operators.

Protocol Buffers
----------------

We need the C++ library for protocol buffers with v3.0.0 or
higher. This is available on some platforms via the package manager
```
brew install --devel protobuf
```
and on others can be built/installed from source
https://github.com/google/protobuf

Installation
------------

Epsilon is pip-installable (currently, this still requires the protocol buffer
dependency, see above).
```
pip install epsilon
```

It can also be pulled from github and installed with
```
python setup.py install
```
