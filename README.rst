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

* numpy
* scipy
* matplotlib
* Neuron: installation instructions here: https://neuron.yale.edu/neuron/ (>=7.7)

Optional dependencies
---------------------

* joblib (for simulating trials simultaneously)
* mpi4py (for simulating the cells in parallel for a single trial). Also depends on:

  * openmpi or other mpi platform installed on system
  * psutil

Installation
============

We recommend the `Anaconda Python distribution <https://www.anaconda.com/products/individual>`_.
To install ``hnn-core``, you first need to install its dependencies::

	$ conda install numpy matplotlib scipy

Additionally, you would need Neuron which is available here:
`https://neuron.yale.edu/neuron/ <https://neuron.yale.edu/neuron/>`_.
We recommend that you install it via pip::

	$ pip install NEURON

Note that if you choose to install Neuron using the traditional installer, it may be hard
to manage it with virtual environments in Anaconda. For this reason, we recommend the
pip installer.

Since ``hnn-core`` does not yet have a stable release, we recommend installing the nightly version. This may change in the future if more users start using it.

To install the latest version of the code (nightly) do::

	$ pip install --upgrade https://api.github.com/repos/jonescompneurolab/hnn-core/zipball/master

To check if everything worked fine, you can do::

	$ python -c 'import hnn_core'

and it should not give any error messages.

Parallel backends
=================

For further instructions on installation and usage of parallel backends for using more than one core, refer to `parallel_backends`_

Bug reports
===========

Use the `github issue tracker <https://github.com/jonescompneurolab/hnn-core/issues>`_ to report bugs.

Interested in Contributing?
===========================

Read our `contributing guide <https://github.com/jonescompneurolab/hnn-core/blob/master/CONTRIBUTING.rst>`_.

.. _parallel_backends: https://jonescompneurolab.github.io/hnn-core/parallel.html
