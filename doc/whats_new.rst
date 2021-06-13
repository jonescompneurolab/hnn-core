:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: hnn_core

.. _0.2:

Current
-------

Changelog
~~~~~~~~~
- Store all connectivity information under :attr:`~hnn_core.Network.connectivity` before building the network, by `Nick Tolley`_ in `#276 <https://github.com/jonescompneurolab/hnn-core/pull/276>`_

- Add new function :func:`~hnn_core.viz.plot_cell_morphology` to visualize cell morphology, by `Mainak Jas`_ in `#319 <https://github.com/jonescompneurolab/hnn-core/pull/319>`_

- Compute dipole component in z-direction automatically from cell morphology instead of hard coding, by `Mainak Jas`_ in  `#327 <https://github.com/jonescompneurolab/hnn-core/pull/320>`_

- Store :class:`~hnn_core.Cell` instances in :class:`~hnn_core.Network`'s :attr:`~/hnn_core.Network.cells` attribute by `Ryan Thorpe`_ in `#321 <https://github.com/jonescompneurolab/hnn-core/pull/321>`_

- Add probability argument to :func:`~hnn_core.Network.add_connection`. Connectivity patterns can also be visualized with :func:`~hnn_core.viz.plot_connectivity_matrix`, by `Nick Tolley`_ in `#318 <https://github.com/jonescompneurolab/hnn-core/pull/318>`_

- Add function to visualize connections originating from individual cells :func:`~hnn_core.viz.plot_cell_connectivity`, by `Nick Tolley`_ in `#339 <https://github.com/jonescompneurolab/hnn-core/pull/339>`_

Bug
~~~

- Remove rounding error caused by repositioning of NEURON cell sections, by `Mainak Jas`_ and `Ryan Thorpe`_ in `#314 <https://github.com/jonescompneurolab/hnn-core/pull/314>`_

- Fix issue where common drives use the same parameters for all cell types, by `Nick Tolley`_ in `#350 <https://github.com/jonescompneurolab/hnn-core/pull/350>`_

API
~~~
- New API for defining cell-cell connections. Custom connections can be added with :func:`~hnn_core.Network.add_connection`, by `Nick Tolley`_ in `#276 <https://github.com/jonescompneurolab/hnn-core/pull/276>`_

- Remove :class:`~hnn_core.L2Pyr`, :class:`~hnn_core.L5Pyr`, :class:`~hnn_core.L2Basket`, and :class:`~hnn_core.L5Basket` classes
  in favor of instantation through functions and a more consistent :class:`~hnn_core.Cell` class by `Mainak Jas`_ in  `#322 <https://github.com/jonescompneurolab/hnn-core/pull/320>`_

- Remove parameter `distribution` in :func:`~hnn_core.Network.add_bursty_drive`. The distribution is now Gaussian by default, by `Mainak Jas`_ in `#330 <https://github.com/jonescompneurolab/hnn-core/pull/330>`_

- New API for accessing and modifying :class:`~hnn_core.Cell` attributes (e.g., synapse and biophysics parameters) as cells are now instantiated from template cells specified
  in a :class:`~hnn_core.Network` instance's :attr:`~/hnn_core.Network.cell_types` attribute by `Ryan Thorpe`_ in `#321 <https://github.com/jonescompneurolab/hnn-core/pull/321>`_

- New API for network creation. The default network is now created with ``net = default_network(params)``, by `Nick Tolley`_ in `#318 <https://github.com/jonescompneurolab/hnn-core/pull/318>`_

- Replace parameter `T` with `tstop` in :func:`~hnn_core.Network.add_tonic_bias` and :func:`~hnn_core.Cell.create_tonic_bias` to be more consistent with other functions and improve readability, by `Kenneth Loi`_ in `#354 <https://github.com/jonescompneurolab/hnn-core/pull/354>`_

.. _0.1:

0.1
---

Changelog
~~~~~~~~~

- Add ability to simulate multiple trials in parallel using joblibs, by `Mainak Jas`_ in `#44 <https://github.com/jonescompneurolab/hnn-core/pull/44>`_

- Rhythmic inputs can now be turned off by setting their conductance weights to 0 instead of setting their start times to exceed the simulation stop time, by `Ryan Thorpe`_ in `#105 <https://github.com/jonescompneurolab/hnn-core/pull/105>`_

- Reader for parameter files, by `Blake Caldwell`_ in `#80 <https://github.com/jonescompneurolab/hnn-core/pull/80>`_

- Add plotting of voltage at soma to inspect firing pattern of cells, by `Mainak Jas`_ in `#86 <https://github.com/jonescompneurolab/hnn-core/pull/86>`_

- Add ability to simulate a single trial in parallel across cores using MPI, by `Blake Caldwell`_ in `#79 <https://github.com/jonescompneurolab/hnn-core/pull/79>`_

- Modify :func:`~hnn_core.viz.plot_dipole` to accept both lists and individual instances of Dipole object, by `Nick Tolley`_ in `#145 <https://github.com/jonescompneurolab/hnn-core/pull/145>`_

- Update ``plot_hist_input`` to :func:`~hnn_core.viz.plot_spikes_hist` which can plot histogram of spikes for any cell type, by `Nick Tolley`_ in `#157 <https://github.com/jonescompneurolab/hnn-core/pull/157>`_

- Add function to compute mean spike rates with user specified calculation type, by `Nick Tolley`_ and `Mainak Jas`_ in `#155 <https://github.com/jonescompneurolab/hnn-core/pull/155>`_

- Add ability to record somatic voltages from all cells, by `Nick Tolley`_ in `#190 <https://github.com/jonescompneurolab/hnn-core/pull/190>`_

- Add ability to instantiate external feed event times of a network prior to building it, by `Christopher Bailey`_ in `#191 <https://github.com/jonescompneurolab/hnn-core/pull/191>`_

- Add ability to record somatic currents from all cells, by `Nick Tolley`_ in `#199 <https://github.com/jonescompneurolab/hnn-core/pull/199>`_

- Add option to turn off dipole postprocessing, by `Carmen Kohl`_ in `#188 <https://github.com/jonescompneurolab/hnn-core/pull/188>`_

- Add ability to add tonic inputs to cell types with :func:`~hnn_core.Network.add_tonic_bias`, by `Mainak Jas`_ in `#209 <https://github.com/jonescompneurolab/hnn-core/pull/209>`_

- Modify :func:`~hnn_core.viz.plot_spikes_raster` to display individual cells, by `Nick Tolley`_ in `#231 <https://github.com/jonescompneurolab/hnn-core/pull/231>`_

- Add :meth:`~hnn_core.Network.copy` method for cloning a ``Network`` instance, by `Christopher Bailey`_ in `#221 <https://github.com/jonescompneurolab/hnn-core/pull/221>`_

- Add methods for creating input drives and biases to network: :meth:`~hnn_core.Network.add_evoked_drive`, :meth:`~hnn_core.Network.add_poisson_drive`, :meth:`~hnn_core.Network.add_bursty_drive` and :meth:`~hnn_core.Network.add_tonic_bias`, by `Christopher Bailey`_ in `#221 <https://github.com/jonescompneurolab/hnn-core/pull/221>`_

- Add functions for plotting power spectral density (:func:`~hnn_core.viz.plot_psd`) and Morlet time-frequency representations (:func:`~hnn_core.viz.plot_tfr_morlet`), by `Christopher Bailey`_ in `#264 <https://github.com/jonescompneurolab/hnn-core/pull/264>`_

- Add y-label units (nAm) to all visualisation functions involving dipole moments, by `Christopher Bailey`_ in `#264 <https://github.com/jonescompneurolab/hnn-core/pull/264>`_

- Add Savitzky-Golay filtering method :meth:`~hnn_core.dipole.Dipole.savgol_filter` to ``Dipole``; copied from ``mne-python`` :meth:`~mne.Evoked.savgol_filter`, by `Christopher Bailey`_ in `#264 <https://github.com/jonescompneurolab/hnn-core/pull/264>`_

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

- Add :func:`~hnn_core.dipole.average_dipoles` function, by `Blake Caldwell`_ in `#156 <https://github.com/jonescompneurolab/hnn-core/pull/156>`_

- New API for defining external drives and biases to network. By default, a :class:`~hnn_core.Network` is created without drives, which are added using class methods. The argument ``add_drives_from_params`` controls this behaviour, by `Christopher Bailey`_ in `#221 <https://github.com/jonescompneurolab/hnn-core/pull/221>`_

- Examples apply random state seeds that reproduce the output of HNN GUI documentation, by `Christopher Bailey`_ in `#221 <https://github.com/jonescompneurolab/hnn-core/pull/221>`_

- Force conversion to nAm (from fAm) for output of :func:`~hnn_core.dipole.simulate_dipole` regardless of ``postproc``-argument, which now only controls parameter file-based smoothing and scaling, by `Christopher Bailey`_ in `#264 <https://github.com/jonescompneurolab/hnn-core/pull/264>`_

People who contributed to this release (in alphabetical order):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Blake Caldwell`_
- `Christopher Bailey`_
- `Carmen Kohl`_
- `Mainak Jas`_
- `Nick Tolley`_
- `Ryan Thorpe`_
- `Samika Kanekar`_
- `Stephanie Jones`_

.. _Blake Caldwell: https://github.com/blakecaldwell
.. _Christopher Bailey: https://github.com/cjayb
.. _Carmen Kohl: https://github.com/kohl-carmen
.. _Kenneth Loi: https://github.com/kenloi
.. _Mainak Jas: http://jasmainak.github.io/
.. _Nick Tolley: https://github.com/ntolley
.. _Ryan Thorpe: https://github.com/rythorpe
.. _Samika Kanekar: https://github.com/samikane
.. _Stephanie Jones: https://github.com/stephanie-r-jones
