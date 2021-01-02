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

- Add function to compute mean spike rates with user specified calculation type, by `Nick Tolley`_ and `Mainak Jas`_ in `#155 <https://github.com/jonescompneurolab/hnn-core/pull/155>`_

- Add ability to record somatic voltages from all cells, by `Nick Tolley`_ in `#190 <https://github.com/jonescompneurolab/hnn-core/pull/190>`_

- Add ability to instantiate external feed event times of a network prior to building it, by `Christopher Bailey`_ in `#191 <https://github.com/jonescompneurolab/hnn-core/pull/191>`_

- Add ability to record somatic currents from all cells, by `Nick Tolley`_ in `#199 <https://github.com/jonescompneurolab/hnn-core/pull/199>`_

- Add option to turn off dipole postprocessing, by `Carmen Kohl`_ in `#188 <https://github.com/jonescompneurolab/hnn-core/pull/188>`_

- Add ability to add tonic inputs to cell types with :func:`hnn_core.Network.add_tonic_input`, by `Mainak Jas`_ in `#209 <https://github.com/jonescompneurolab/hnn-core/pull/209>`_

- Modify :func:`hnn_core.viz.plot_spikes_raster` to display individual cells, by `Nick Tolley`_ in `#231 <https://github.com/jonescompneurolab/hnn-core/pull/231>`_

- Add methods for creating input drives and biases to network: :func:`hnn_core.Network.add_evoked_drive`, :func:`hnn_core.Network.add_gaussian_drive`, :func:`hnn_core.Network.add_poisson_drive`, :func:`hnn_core.Network.add_bursty_drive` and :func:`hnn_core.Network.add_tonic_bias`, by `Christopher Bailey`_ in `#221 <https://github.com/jonescompneurolab/hnn-core/pull/221>`_

Bug
~~~

- Fix missing autapses in network construction, by `Mainak Jas`_ in `#50 <https://github.com/jonescompneurolab/hnn-core/pull/50>`_

- Fix rhythmic input feed, by `Ryan Thorpe`_ in `#98 <https://github.com/jonescompneurolab/hnn-core/pull/98>`_

- Fix bug introduced into rhythmic input feed and add test, by `Christopher Bailey`_ in `#102 <https://github.com/jonescompneurolab/hnn-core/pull/102>`_

- Fix bug in amplitude of delay (for connection between L2 Basket and Gaussian feed) being passed incorrectly, by `Mainak Jas`_ in `#146 <https://github.com/jonescompneurolab/hnn-core/pull/146>`_

- Connections now cannot be removed by setting the weights to 0., by `Mainak Jas`_ and `Ryan Thorpe`_ in `#162 <https://github.com/jonescompneurolab/hnn-core/pull/162>`_

- MPI and Joblib backends now apply jitter across multiple trials identically, by `Ryan Thorpe`_ in `#171 <https://github.com/jonescompneurolab/hnn-core/pull/171>`_

- Fix bug in Poisson input where the first spike was being missed after the start time, by `Mainak Jas`_ in `#204 <https://github.com/jonescompneurolab/hnn-core/pull/204/>`_

- Fix bug in network to add empty spike when empty file is read in, by `Samika Kanekar`_ and `Ryan Thorpe`_ in `#207 <https://github.com/jonescompneurolab/hnn-core/pull/207>`_

API
~~~

- Make a context manager for Network class, by `Mainak Jas`_ and `Blake Caldwell`_ in `#86 <https://github.com/jonescompneurolab/hnn-core/pull/86>`_

- Create Spikes class, add write methods and read functions for Spikes and Dipole classes, by `Ryan Thorpe`_ in `#96 <https://github.com/jonescompneurolab/hnn-core/pull/96>`_

- Only specify `n_jobs` when instantiating the JoblibBackend, by `Blake Caldwell`_ in `#79 <https://github.com/jonescompneurolab/hnn-core/pull/79>`_

- Make a context manager for parallel backends (JoblibBackend, MPIBackend), by `Blake Caldwell`_ in `#79 <https://github.com/jonescompneurolab/hnn-core/pull/79>`_

- Add average_dipoles function to `hnn_core.dipole`, by `Blake Caldwell`_ in `#156 <https://github.com/jonescompneurolab/hnn-core/pull/156>`_

- New API for defining external drives and biases to network, by `Christopher Bailey`_ in `#221 <https://github.com/jonescompneurolab/hnn-core/pull/221>`_

.. _Mainak Jas: http://jasmainak.github.io/
.. _Blake Caldwell: https://github.com/blakecaldwell
.. _Ryan Thorpe: https://github.com/rythorpe
.. _Christopher Bailey: https://github.com/cjayb
.. _Nick Tolley: https://github.com/ntolley
.. _Carmen Kohl: https://github.com/kohl-carmen
.. _Samika Kanekar: https://github.com/samikane
