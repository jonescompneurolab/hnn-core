:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: hnn_core

.. _current:

Current
-------

Changelog
~~~~~~~~~

- Add ability to simulate multiple trials in parallel using joblibs, by `Mainak Jas`_ in `#44 <https://github.com/jonescompneurolab/hnn-core/pull/44>`_

- Rhythmic inputs can now be turned off by setting their conductance weights to 0 instead of setting their start times to exceed the simulation stop time, by `Ryan Thorpe`_ in `#105 <https://github.com/jonescompneurolab/hnn-core/pull/105>`_

- Reader for parameter files, by `Blake Caldwell`_ in `#80 <https://github.com/jonescompneurolab/hnn-core/pull/80>`_

- Add plotting of voltage at soma to inspect firing pattern of cells, by `Mainak Jas`_ in `#86 <https://github.com/jonescompneurolab/hnn-core/pull/86>`_

- Add ability to simulate a single trial in parallel across cores using MPI, by `Blake Caldwell`_ in `#79 <https://github.com/jonescompneurolab/hnn-core/pull/79>`_

- Modify plot_dipole() to accept both lists and individual instances of Dipole object, by `Nick Tolley`_ in `#145 <https://github.com/jonescompneurolab/hnn-core/pull/145>`_

- Update plot_hist_input() to plot_spikes_hist() which can plot histogram of spikes for any cell type, by `Nick Tolley`_ in `#157 <https://github.com/jonescompneurolab/hnn-core/pull/157>`_

Bug
~~~

- Fix missing autapses in network construction, by `Mainak Jas`_ in `#50 <https://github.com/jonescompneurolab/hnn-core/pull/50>`_

- Fix rhythmic input feed, by `Ryan Thorpe`_ in `#98 <https://github.com/jonescompneurolab/hnn-core/pull/98>`_

- Fix bug introduced into rhythmic input feed and add test, by `Christopher Bailey`_ in `#102 <https://github.com/jonescompneurolab/hnn-core/pull/102>`_

- Fix bug in amplitude of delay (for connection between L2 Basket and Gaussian feed) being passed incorrectly, by `Mainak Jas`_ in `#146 <https://github.com/jonescompneurolab/hnn-core/pull/146>`_

- Connections now cannot be removed by setting the weights to 0., by `Mainak Jas`_ and `Ryan Thorpe`_ in `#162 <https://github.com/jonescompneurolab/hnn-core/pull/162>`_

API
~~~

- Make a context manager for Network class, by `Mainak Jas`_ and `Blake Caldwell`_ in `#86 <https://github.com/jonescompneurolab/hnn-core/pull/86>`_

- Create Spikes class, add write methods and read functions for Spikes and Dipole classes, by `Ryan Thorpe`_ in `#96 <https://github.com/jonescompneurolab/hnn-core/pull/96>`_

- Only specify `n_jobs` when instantiating the JoblibBackend, by `Blake Caldwell`_ in `#79 <https://github.com/jonescompneurolab/hnn-core/pull/79>`_

- Make a context manager for parallel backends (JoblibBackend, MPIBackend), by `Blake Caldwell`_ in `#79 <https://github.com/jonescompneurolab/hnn-core/pull/79>`_

- Add average_dipoles function to `hnn_core.dipole`, by `Blake Caldwell`_ in `#156 <https://github.com/jonescompneurolab/hnn-core/pull/156>`_

.. _Mainak Jas: http://jasmainak.github.io/
.. _Blake Caldwell: https://github.com/blakecaldwell
.. _Ryan Thorpe: https://github.com/rythorpe
.. _Christopher Bailey: https://github.com/cjayb
.. _Nick Tolley: https://github.com/ntolley
