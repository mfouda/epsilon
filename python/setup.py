from setuptools import setup, Extension

solve = Extension(
    name = "epsilon/solve",
    sources = ["epsilon/solvemodule.cc"],
    extra_compile_args = ["--std=c++14"]
)

setup(
    name = "epsilon",
    version = "1.0a1",
    author = "Matt Wytock",
    author_email = "mwytock@gmail.com",
    packages = ["epsilon"],
    install_requires = ["cvxpy"],
    ext_modules = [solve],
)
