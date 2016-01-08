#!/bin/bash -eu

python setup.py bdist_wheel
aws s3 cp dist/*.whl s3://epopt/
