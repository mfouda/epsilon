"""Setup script for epsilon.

Epsilon depends on having protobuf >3.0.0a3 installed. We assume that `protoc`
is avaiable on the command line and that the library/headers live in the same
location.
"""

import os
import subprocess
import sys

from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build_py import build_py as _build_py
from distutils.spawn import find_executable
from setuptools import find_packages, setup, Extension

SOURCE_DIR = "../src"
PROTO_DIR = "../proto"
EIGEN_DIR = "../third_party/eigen"

PROTOC = find_executable("protoc")
if PROTOC is None:
    std.stderr.write("protoc is not installed")
    sys.exit(-1)
PROTOBUF_PREFIX = os.path.dirname(os.path.dirname(PROTOC))
PROTOBUF_LIB = os.path.join(PROTOBUF_PREFIX, "lib")

EPSILON_SOURCES = [
    "epsilon/algorithms/prox_admm.cc",
    "epsilon/algorithms/solver.cc",
    "epsilon/expression/expression.cc",
    "epsilon/expression/problem.cc",
    "epsilon/file/file.cc",
    "epsilon/operators/affine.cc",
    "epsilon/operators/prox.cc",
    "epsilon/parameters/local_parameter_service.cc",
    "epsilon/util/dynamic_matrix.cc",
    "epsilon/util/string.cc",
    "epsilon/util/time.cc",
    "epsilon/util/vector.cc",
    "epsilon/util/vector_file.cc",
]
EPSILON_SOURCES = [os.path.abspath(SOURCE_DIR + "/" + src)
                   for src in EPSILON_SOURCES]

EPSILON_PROTOS = [
    "epsilon/data.proto",
    "epsilon/expression.proto",
    "epsilon/prox.proto",
    "epsilon/solver_params.proto",
    "epsilon/stats.proto",
    "epsilon/status.proto",
]
EPSILON_PROTO_SOURCES = [src.replace(".proto", ".pb.cc")
                         for src in EPSILON_PROTOS]


PROTO_PY = 0
PROTO_CC = 1
def generate_proto(source, output):
    source_path = os.path.join(PROTO_DIR, source)
    if output == PROTO_PY:
        out_path = source.replace(".proto", "_pb2.py")
        out_flag = "--python_out=."
    else:
        out_path = source.replace(".proto", "pb.cc")
        out_flag = "--cpp_out=."

    if (os.path.isfile(out_path) and
        os.path.getmtime(out_path) > os.path.getmtime(source_path)):
        return

    print "Generating %s..." % out_path
    cmd = [PROTOC, "-I", PROTO_DIR, source_path, out_flag]
    if subprocess.call(cmd) != 0:
        sys.exit(-1)

class build_py(_build_py):
    def run(self):
        for proto in EPSILON_PROTOS:
            generate_proto(proto, PROTO_PY)
        _build_py.run(self)

class build_ext(_build_ext):
    def run(self):
        for proto in EPSILON_PROTOS:
            generate_proto(proto, PROTO_CC)
        _build_ext.run(self)

solve = Extension(
    "epsilon._solve",
    language = "c++",
    sources = (
        EPSILON_PROTO_SOURCES +
        EPSILON_SOURCES +
        ["epsilon/solvemodule.cc"]),
    extra_compile_args = ["-std=c++14"],
    include_dirs = [
        SOURCE_DIR,
        EIGEN_DIR,
        PROTOBUF_LIB,
        ".",
    ],
    library_dirs = [
        PROTOBUF_LIB
    ],
    libraries = ["protobuf"],
)

setup(
    name = "epsilon",
    version = "1.0a1",
    author = "Matt Wytock",
    author_email = "mwytock@gmail.com",
    packages = find_packages(),
    ext_modules = [solve],
    install_requires = [
        "cvxpy==0.2.28",
        "protobuf==3.0.0a3"
    ],
    cmdclass = {
        "build_ext": build_ext,
        "build_py": build_py,
    },
)
