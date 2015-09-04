from setuptools import setup

setup(
    name="distopt",
    version="1.0a1",
    author="Matt Wytock",
    author_email="mwytock@gmail.com",
    packages=["distopt"],
    install_requires=["cvxpy", "grpcio", "boto"],
)
