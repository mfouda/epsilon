"""Setup script for epsilon.

Epsilon depends on having protobuf >3.0.0a3 installed. We assume that `protoc`
is avaiable on the command line and that the library/headers live in the same
location.
"""

import os
import subprocess
import sys

from distutils.command.build_ext import build_ext
from distutils.command.build_py import build_py
from distutils.dep_util import newer
from distutils.spawn import find_executable
from distutils import log
from setuptools import find_packages, setup, Extension, Command

SOURCE_DIR = "src"
PROTO_DIR = "proto"
EIGEN_DIR = "third_party/eigen"

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
EPSILON_SOURCES = [SOURCE_DIR + "/" + src for src in EPSILON_SOURCES]

EPSILON_PROTOS = [
    "epsilon/data.proto",
    "epsilon/expression.proto",
    "epsilon/prox.proto",
    "epsilon/solver_params.proto",
    "epsilon/stats.proto",
    "epsilon/status.proto",
]

PROTO_PY = 0
PROTO_CC = 1
def generate_proto(src_name, format, dst_dir, verbose):
    src = os.path.join(PROTO_DIR, src_name)
    if format == PROTO_PY:
        dst = os.path.join(dst_dir, src_name.replace(".proto", "_pb2.py"))
        out_flag = "--python_out=" + dst_dir
    else:
        dst = os.path.join(dst_dir, src_name.replace(".proto", ".pb.cc"))
        out_flag = "--cpp_out=" + dst_dir

    if newer(src, dst):
        if verbose >= 1:
            log.debug("generating %s...", dst)

        cmd = [PROTOC, "-I", PROTO_DIR, src, out_flag]
        if subprocess.call(cmd) != 0:
            sys.exit(-1)

    return dst

class BuildPyCommand(build_py):
    def run(self):
        self.mkpath(self.build_lib)
        for proto in EPSILON_PROTOS:
            generate_proto(proto, PROTO_PY, self.build_lib, self.verbose)
        build_py.run(self)

class BuildExtCommand(build_ext):
    def run(self):
        for ext in self.extensions:
            ext.sources = self.gen_proto_sources(ext.sources, ext)
        build_ext.run(self)

    def gen_proto_sources(self, sources, ext):
        new_sources = []
        has_proto = False
        for source in sources:
            if os.path.splitext(source)[1] == ".proto":
                has_proto = True
                self.mkpath(self.build_temp)
                new_sources.append(
                    generate_proto(
                        source, PROTO_CC, self.build_temp, self.verbose))
            else:
                new_sources.append(source)

        if has_proto:
            ext.include_dirs.append(self.build_temp)

        return new_sources

class CleanCommand(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system("rm -rf ./build ./dist ./*.egg-info")

solve = Extension(
    "epsilon._solve",
    language = "c++",
    sources = (
        EPSILON_PROTOS +
        EPSILON_SOURCES +
        ["python/epsilon/solvemodule.cc"]),
    extra_compile_args = ["-std=c++14"],
    include_dirs = [
        SOURCE_DIR,
        EIGEN_DIR,
        PROTOBUF_LIB,
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
    url = "https://github.com/mwytock/epsilon",
    author_email = "mwytock@gmail.com",
    packages = find_packages("python"),
    package_dir = {"epsilon": "python/epsilon"},
    ext_modules = [solve],
    install_requires = [
        "cvxpy==0.2.28",
        "protobuf==3.0.0a3"
    ],
    cmdclass = {
        "build_ext": BuildExtCommand,
        "build_py": BuildPyCommand,
        "clean": CleanCommand
    },
)
