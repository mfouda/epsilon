#!/bin/bash -eux
#
# Script to run tests on Travis CI

# Numpy/scipy environment via conda
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
conda install --yes python=2.7 pip atlas numpy scipy nose

# Environment information
ls -l /home/travis/miniconda/lib
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# Install cvxpy
pip install cvxpy

# Install epsilon
cd python
python setup.py install
