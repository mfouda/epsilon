Quick Start
===========

The current version of Epsilon is built on top of CVXPY and thus getting started
amounts to `installing CVXPY
<http://www.cvxpy.org/en/latest/install/index.html>`_ followed by the the ``epopt`` Python
package.

Prerequisites
-------------

Epsilon requires the basic numerical python environment (NumPy/SciPy) which is
also required for CVXPY. More `detailed instructions
<http://www.cvxpy.org/en/latest/install/index.html>`_ are available from the
CVXPY website, in essence you need to first install the latest version of
NumPy/SciPy:

.. code:: bash

   pip install -U numpy scipy
   pip install -U cvxpy
..

Binary installation
-------------------

Once the numerical python environment is installed, the easiest way to get
started with Epsilon is by installing a binary distribution via ``pip``.

Mac OS X
~~~~~~~~

.. code:: bash

   pip install -U http://epopt.s3.amazonaws.com/epopt-0.3.1-cp27-none-macosx_10_11_x86_64.whl
..

Linux
~~~~~

.. code:: bash

   pip install -U http://epopt.s3.amazonaws.com/epopt-0.3.1-cp27-none-linux_x86_64.whl
..

Source installation
-------------------

For development purposes, Epsilon can also be downloaded and compiled
directly from the github repository.

Build third-party C++ dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, compile the third-party C++ dependencies (gflags, glog, protobuf) which
are provided as git submodules linked from the main repository using the provide
script.

.. code:: bash

   git submodule update --init
   tools/build_third_party.sh
..

This script has been tested on recent versions of Mac OS X and
Debian/Ubuntu. Note that third-party dependencies may requires some additional
libraries which can be installed using the appropriate package manager,
e.g. homebrew on OS X or ``apt-get`` on Linux.

Build Epsilon and run tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile the C++ code and run tests

.. code:: bash

   make -j test
..

Now build the C++ Python extension and set up the local development environment

.. code:: bash

   python setup.py build
   python setup.py -q develop --user
..

Finally, run the python tests using ``nose``

.. code:: bash

   pip install nose
   nosetests epopt
..
