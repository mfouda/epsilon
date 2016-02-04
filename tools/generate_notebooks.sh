#!/bin/bash -eu

notebooks=$PWD/notebooks/*.ipynb
output=docs/notebooks

cd $output
for nb in $notebooks; do
    ipython nbconvert $nb --to rst
done
