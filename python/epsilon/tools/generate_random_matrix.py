#!/usr/bin/env python
"""Simple tool to generate a random matrix."""

import logging
import argparse

import numpy

import distopt

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("m", type=int)
parser.add_argument("n", type=int)
parser.add_argument("prefix")
args = parser.parse_args()

def main():
    logging.info("Generating %d x %d randn matrix ...", args.m, args.n)
    seed = (abs(hash(args.prefix))) % 2**32
    numpy.random.seed(abs(hash(args.prefix)) % 2**32)
    A = numpy.random.randn(args.m, args.n)
    distopt.write_matrix(args.prefix, A)

if __name__ == "__main__":
    main()
