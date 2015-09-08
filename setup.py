"""Setup script for epsilon."""

import os
import subprocess
import sys

from setuptools import setup, Extension

solve = Extension(
    name = "epsilon._solve",
    sources = ["python/epsilon/solvemodule.cc"],
    language = "c++",
    extra_compile_args = ["-std=c++14"],
    extra_objects = ["build-cc/libepsilon.a"],
    include_dirs = [
        "/usr/local/include",
        "build-cc",
        "src",
        "third_party/eigen",
    ],
    library_dirs = [
        "/usr/local/lib",
    ],
    libraries = ["protobuf"],
)

setup(
    name = "epsilon",
    version = "1.0a1",
    author = "Matt Wytock",
    url = "https://github.com/mwytock/epsilon",
    author_email = "mwytock@gmail.com",
    packages = ["epsilon"],
    package_dir = {"epsilon": "python/epsilon"},
    ext_modules = [solve],
    install_requires = [
        "cvxpy==0.2.28",
        "protobuf==3.0.0a3"
    ],
)
