hnn-core
========

|tests| |CircleCI| |Codecov| |PyPI| |Gitter|

|HNN-GUI|

This is a leaner and cleaner version of the code based off the `HNN repository <https://github.com/jonescompneurolab/hnn>`_.

It is early Work in Progress. Contributors are very welcome.

Dependencies
------------

* numpy
* scipy
* matplotlib
* Neuron (>=7.7)

Optional dependencies
---------------------

GUI
~~~

* ipywidgets
* voila

Parallel processing
~~~~~~~~~~~~~~~~~~~

* joblib (for simulating trials simultaneously)
* mpi4py (for simulating the cells in parallel for a single trial). Also depends on:

  * openmpi or other mpi platform installed on system
  * psutil

Installation
============

We recommend the `Anaconda Python distribution <https://www.anaconda.com/products/individual>`_.
To install ``hnn-core``, simply do::

   $ pip install hnn_core

and it will install ``hnn-core`` along with the dependencies which are not already installed.

Note that if you installed Neuron using the traditional installer package, it is recommended
to remove it first and unset ``PYTHONPATH`` and ``PYTHONHOME`` if they were set. This is
because the pip installer works better with virtual environments such as the ones provided by ``conda``.

If you want to track the latest developments of ``hnn-core``, you can install the current version of the code (nightly) with::

	$ pip install --upgrade https://api.github.com/repos/jonescompneurolab/hnn-core/zipball/master

To check if everything worked fine, you can do::

	$ python -c 'import hnn_core'

and it should not give any error messages.

**GUI installation**

To install the GUI dependencies along with ``hnn-core``, a simple tweak to the above command is needed::

   $ pip install hnn_core[gui]

To start the GUI, please do::

   $ hnn-gui

**Parallel backends**

For further instructions on installation and usage of parallel backends for using more
than one CPU core, refer to our `parallel backend guide`_.

**Note for Windows users**

The pip installer for Neuron does not yet work for Windows. In this case, it is better to
install ``hnn_core`` without the dependencies::

   $ pip install hnn_core --no-deps

and then install the dependencies separately::

   $ pip install scipy numpy matplotlib

and install Neuron using the traditional package installer available here
`https://neuron.yale.edu/neuron/ <https://neuron.yale.edu/neuron/>`_.

Documentation and examples
==========================

Once you have tested that ``hnn_core`` and its dependencies were installed,
we recommend downloading and executing the
`example scripts <https://jonescompneurolab.github.io/hnn-core/stable/auto_examples/index.html>`_
provided on the `documentation pages <https://jonescompneurolab.github.io/hnn-core/>`_
(as well as in the `GitHub repository <https://github.com/jonescompneurolab/hnn-core>`_).

Note that ``python`` plots are by default non-interactive (blocking): each plot must thus be closed before the code execution continues. We recommend using and 'interactive' python interpreter such as ``ipython``::

   $ ipython --matplotlib

and executing the scripts using the ``%run``-magic::

   %run plot_simulate_evoked.py

When executed in this manner, the scripts will execute entirely, after which all plots will be shown. For an even more interactive experience, in which you execute code and interrogate plots in sequential blocks, we recommend editors such as `VS Code <https://code.visualstudio.com>`_ and `Spyder <https://docs.spyder-ide.org/current/index.html>`_.

Bug reports
===========

Use the `github issue tracker <https://github.com/jonescompneurolab/hnn-core/issues>`_ to
report bugs. For user questions and scientific discussions, please join the
`HNN Google group <https://groups.google.com/g/hnnsolver>`_.

Interested in Contributing?
===========================

Read our `contributing guide`_.

.. _parallel backend guide: https://jonescompneurolab.github.io/hnn-core/dev/parallel.html
.. _contributing guide: https://jonescompneurolab.github.io/hnn-core/dev/contributing.html

.. |tests| image:: https://github.com/jonescompneurolab/hnn-core/actions/workflows/unit_tests.yml/badge.svg?branch=master
   :target: https://github.com/jonescompneurolab/hnn-core/actions/?query=branch:master+event:push

.. |CircleCI| image:: https://circleci.com/gh/jonescompneurolab/hnn-core.svg?style=svg
   :target: https://circleci.com/gh/jonescompneurolab/hnn-core

.. |Codecov| image:: https://codecov.io/gh/jonescompneurolab/hnn-core/branch/master/graph/badge.svg
	:target: https://codecov.io/gh/jonescompneurolab/hnn-core

.. |PyPI| image:: https://img.shields.io/pypi/dm/hnn-core.svg?label=PyPI%20downloads
	:target: https://pypi.org/project/hnn-core/

.. |HNN-GUI| image:: https://user-images.githubusercontent.com/11160442/178095018-35d2619a-6a82-4e27-91c9-ff2796fab435.png

.. |Gitter| image:: https://badges.gitter.im/jonescompneurolab/hnn_core.svg
   :target: https://gitter.im/jonescompneurolab/hnn-core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge