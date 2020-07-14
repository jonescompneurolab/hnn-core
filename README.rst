hnn-core
========

.. image:: https://badges.gitter.im/hnn-core/hnn-core.svg
   :target: https://gitter.im/hnn-core/hnn-core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Gitter

.. image:: https://circleci.com/gh/jonescompneurolab/hnn-core.svg?style=svg
   :target: https://circleci.com/gh/jonescompneurolab/hnn-core
   :alt: CircleCi

.. image:: https://api.travis-ci.org/jonescompneurolab/hnn-core.svg?branch=master
    :target: https://travis-ci.org/jonescompneurolab/hnn-core
    :alt: Build Status

.. image:: https://codecov.io/gh/jonescompneurolab/hnn-core/branch/master/graph/badge.svg
	:target: https://codecov.io/gh/jonescompneurolab/hnn-core
	:alt: Test coverage

This is a leaner and cleaner version of the code based off the `HNN repository <https://github.com/jonescompneurolab/hnn>`_. However, a Graphical User Interface is not supported at the moment in this repository.

It is early Work in Progress. Contributors are very welcome.

Dependencies
------------

* Neuron: installation instructions here: https://neuron.yale.edu/neuron/
* scipy
* numpy
* matplotlib
* joblib (optional for running trials simultaneously)
* mpi4py (optional for running each trial in parallel across cores). Also depends on:

  * openmpi or other mpi platform installed on system
  * psutil

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``hnn-core``, you first need to install its dependencies::

	$ conda install numpy matplotlib scipy

For using more than one CPU core, see :ref:`Parallel backends` below.

Additionally, you would need Neuron which is available here: `https://neuron.yale.edu/neuron/ <https://neuron.yale.edu/neuron/>`_

Since ``hnn-core`` does not yet have a stable release, we recommend installing the nightly version. This may change in the future if more users start using it.

To install the latest version of the code (nightly) do::

	$ git clone https://github.com/jonescompneurolab/hnn-core.git
	$ cd hnn-core/
	$ python setup.py develop

A final step to the installation process is to compile custom ionic channel
mechanisms using `nrnivmodl` from Neuron. To do this, simple do::

	$ cd mod/ && nrnivmodl

inside the ``hnn-core`` directory. It should create the compiled custom mechanism files.

To check if everything worked fine, you can do::

	$ python -c 'import hnn_core'

and it should not give any error messages.

.. _Parallel backends:

Parallel backends
=================

Two options are available for making use of multiple CPU cores. The first runs multiple trials in parallel with joblib. Alternatively, you can run each trial across multiple cores to reduce the runtime.

Joblib
------

This is the default backend and will execute multiple trials at the same time, with each trial running on a separate core in "embarrassingly parallel" execution. Note that with only 1 trial, there will be no parallelism.

**Dependencies**::

	$ pip install joblib

**Usage**::

	from hnn_core import JoblibBackend

	# set n_jobs to the number of trials to run in parallel with Joblib (up to number of cores on system)
	with JoblibBackend(n_jobs=2):
		dpls = simulate_dipole(net, n_trials=2)

MPI
------

This backend will use MPI (Message Passing Interface) on the system to split neurons across CPU cores (processors) and reduce the simulation time as more cores are used.

**Linux Dependencies**::

	$ sudo apt-get install libopenmpi-dev openmpi-bin
	$ pip install mpi4py psutil

**MacOS Dependencies**::

	$ conda install yes openmpi mpi4py
	$ pip install psutil

**MacOS Environment**::

	$ export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

Alternatively, run the commands below will avoid needing to run the export command every time a new shell is opened::

	cd ${CONDA_PREFIX}
	mkdir -p etc/conda/activate.d etc/conda/deactivate.d
	echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> etc/conda/activate.d/env_vars.sh
	echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> etc/conda/activate.d/env_vars.sh
	echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh
	echo "unset OLD_LD_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh

**Test MPI**::

	$ mpiexec -np 2 nrniv -mpi -python -c 'from neuron import h; from mpi4py import MPI; \
	                                       print("Hello from proc %d" % MPI.COMM_WORLD.Get_rank()); \
                                               h.quit()'
	numprocs=2
	NEURON -- VERSION 7.7.2 7.7 (2b7985ba) 2019-06-20
	Duke, Yale, and the BlueBrain Project -- Copyright 1984-2018
	See http://neuron.yale.edu/neuron/credits

	Hello from proc 0
	Hello from proc 1

Verifies that MPI, NEURON, and Python are all working together.

**Usage**::

	from hnn_core import MPIBackend

	# set n_procs to the number of processors MPI can use (up to number of cores on system)
	with MPIBackend(n_procs=2):
		dpls = simulate_dipole(net, n_trials=1)

Bug reports
===========

Use the `github issue tracker <https://github.com/jonescompneurolab/hnn-core/issues>`_ to report bugs.
