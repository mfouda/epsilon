import os

from setuptools import find_packages, setup, Extension

SOURCE_DIR = "../src"
EIGEN_DIR = "../third_party/eigen"

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

solve = Extension(
    "epsilon._solve",
    language = "c++",
    sources = EPSILON_SOURCES + ["epsilon/solvemodule.cc"],
    extra_compile_args = ["-std=c++14"],
    include_dirs = [
        SOURCE_DIR,
        EIGEN_DIR,
        "/usr/local/include",
    ],
    libraries = ["protobuf"],
    library_dirs = [
        "/usr/local/lib",
    ]
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
)
