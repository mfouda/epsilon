#!/usr/bin/env python

import argparse
import os

import numpy as np
import scipy.sparse as sp

import distopt

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("M", type=int)
parser.add_argument("--prefix", default="/s3/us-west-2/distopt/lasso_chunked")
parser.add_argument("--m", default=1000, type=int)
parser.add_argument("--n", default=10000, type=int)
parser.add_argument("--nnzs", default=100, type=int)
parser.add_argument("--sigma", default=1e-3, type=float)
args = parser.parse_args()

np.random.seed(0)

m = args.M*args.m
n = args.n

x0 = sp.rand(n,1,float(args.nnzs)/n)
x0.data = np.random.randn(x0.nnz)
x0 = x0.todense()

A = np.random.randn(m, n)
A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])
b = A*x0 + np.sqrt(args.sigma)*np.random.randn(m,1)

def filename(file_str):
    return os.path.join(args.prefix, args.name, file_str)

print "||Atb||_inf =", np.max(np.abs(A.T.dot(b)))
for i in range(args.M):
    idx = slice(i*args.m, (i+1)*args.m)
    print idx
    distopt.write_matrix(filename("A%d" % i), A[idx, :])
    distopt.write_matrix(filename("b%d" % i), b[idx])
