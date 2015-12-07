"""Setup script for epopt python package."""

import fnmatch
import os
import platform
import subprocess
import sys

from distutils import log
from distutils.dep_util import newer
from distutils.spawn import find_executable
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

BUILD_CC_DIR = "build-cc"
PROTO_DIR = "proto"
PYTHON_DIR = "python"
PYTHON_PROTO_DIR = os.path.join(PYTHON_DIR, "epopt", "proto")
THIRD_PARTY_DIR = os.path.join(BUILD_CC_DIR, "third_party")

PROTOC = find_executable("protoc")
if PROTOC is None:
    std.stderr.write("protoc is not installed")
    sys.exit(-1)

class BuildPyCommand(build_py):
    def run(self):
        self.generate_protos(PROTO_DIR, PYTHON_PROTO_DIR)
        build_py.run(self)

    def generate_protos(self, src_dir, dst_dir):
        for root, dirnames, filenames in os.walk(src_dir):
            for filename in fnmatch.filter(filenames, "*.proto"):
                src_name = os.path.join(root[len(src_dir)+1:], filename)
                self.generate_proto(src_dir, src_name, dst_dir)

    def generate_proto(self, src_dir, src_name, dst_dir):
        src = os.path.join(src_dir, src_name)
        dst = os.path.join(dst_dir, src_name.replace(".proto", "_pb2.py"))

        if not newer(src, dst):
            return

        if self.verbose >= 1:
            log.info("generating %s", dst)

        cmd = [PROTOC, "-I", PROTO_DIR, src, "--python_out=" + dst_dir]
        subprocess.check_call(cmd)

class BuildExtCommand(build_ext):
    def run(self):
        self.make()
        build_ext.run(self)

    def make(self):
        subprocess.check_call("make")

class CleanCommand(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd = ("rm -rf " +
               "./build " +
               "./build-cc " +
               "./dist " +
               "./python/*.egg-info " +
               "./python/epopt/*.pyc " +
               "./python/epopt/*.so " +
               "./python/epopt/compiler/*.pyc " +
               "./python/epopt/problems/*.pyc " +
               "./python/epopt/proto/epsilon/*_pb2.py*")
        subprocess.check_call(cmd, shell=True)

solve_libs = [
    os.path.join(THIRD_PARTY_DIR, "lib", "libprotobuf.a"),
    os.path.join(THIRD_PARTY_DIR, "lib", "libglog.a"),
    os.path.join(THIRD_PARTY_DIR, "lib", "libgflags.a"),
]

epsilon_lib = os.path.join(BUILD_CC_DIR, "libepsilon.a")

solve = Extension(
    name = "epopt._solve",
    sources = ["python/epopt/solvemodule.cc"],
    language = "c++",
    extra_compile_args = ["-std=c++14"],
    depends = [epsilon_lib],
    include_dirs = [
        os.path.join(THIRD_PARTY_DIR, "include"),
        BUILD_CC_DIR,
        "src",
        "third_party/eigen",
    ]
)

# NOTE(mwytock): The -all_load and -Wl,--whole-archive linker flags are needed
# to pull in all symbols from libepsilon.a because these include things that are
# used indirectly via registration (e.g. the proximal operator library)
if platform.system() == "Darwin":
    solve.extra_link_args += solve_libs + [
        "-all_load",
        epsilon_lib]
else:
    solve.extra_link_args += [
        "-Wl,--whole-archive",
        epsilon_lib,
        "-Wl,--no-whole-archive"] + solve_libs

setup(
    name = "epopt",
    version = "0.1.0",
    author = "Matt Wytock",
    url = "http://epopt.io/",
    author_email = "mwytock@gmail.com",
    packages = find_packages(PYTHON_DIR),
    package_dir = {"": PYTHON_DIR},
    package_data = {"": [
        "problems/baby.jpg",
        "problems/mnist_small.mat",
        "problems/mnist_tiny.mat",
    ]},
    ext_modules = [solve],
    install_requires = [
        "cvxpy==0.3.1",
        "protobuf==3.0.0a3"
    ],
    cmdclass = {
        "build_ext": BuildExtCommand,
        "build_py": BuildPyCommand,
        "clean": CleanCommand
    }
)
