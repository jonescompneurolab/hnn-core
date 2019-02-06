.. mne-neuron documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mne-neuron
==========

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``mne-neuron``, you first need to install its dependencies::

	$ conda install numpy matplotlib scipy

Additionally, you would need Neuron which is available here: `https://neuron.yale.edu/neuron/ <https://neuron.yale.edu/neuron/>`_
If you want to install the latest version of the code (nightly) use::

	$ pip install https://api.github.com/repos/jasmainak/mne-neuron/zipball/master

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import mne-neuron'

and it should not give any error messages.

Bug reports
===========

Use the `github issue tracker <https://github.com/jasmainak/mne-neuron/issues>`_ to report bugs.
