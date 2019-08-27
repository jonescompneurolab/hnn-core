hnn-core
========

.. image:: https://badges.gitter.im/hnn-core/hnn-core.svg
   :target: https://gitter.im/hnn-core/hnn-core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Gitter

.. image:: https://api.travis-ci.org/hnnsolver/hnn-core.svg?branch=master
    :target: https://travis-ci.org/hnnsolver/hnn-core
    :alt: Build Status

.. image:: https://codecov.io/gh/hnnsolver/hnn-core/branch/master/graph/badge.svg
	:target: https://codecov.io/gh/hnnsolver/hnn-core
	:alt: Test coverage

This is a leaner and cleaner version of the code based off the `HNN repository <https://github.com/jonescompneurolab/hnn>`_. However, a Graphical User Interface is not supported at the moment in this repository.

It is early Work in Progress. Contributors are very welcome.

Dependencies
------------

* Neuron: installation instructions here: https://neuron.yale.edu/neuron/
* scipy
* numpy
* matplotlib
* joblib (optional for parallel processing)

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``hnn-core``, you first need to install its dependencies::

	$ conda install numpy matplotlib scipy

For joblib, you can do::

	$ pip install joblib

Additionally, you would need Neuron which is available here: `https://neuron.yale.edu/neuron/ <https://neuron.yale.edu/neuron/>`_

If you want to install the latest version of the code (nightly) use::

	$ pip install https://api.github.com/repos/hnnsolver/hnn-core/zipball/master

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import hnn_core'

and it should not give any error messages.

A final step to the installation process is to compile custom ionic channel
mechanisms using `nrnivmodl` from Neuron. To do this, simple do::

	$ make

It should create a directory with the compiled mechanisms.

Bug reports
===========

Use the `github issue tracker <https://github.com/hnnsolver/hnn-core/issues>`_ to report bugs.
