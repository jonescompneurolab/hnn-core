:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: hnn_core

Current
-------

Changelog
~~~~~~~~~
- Add option to select drives using argument 'which_drives' in
  :func:`~hnn_core.optimization.optimize_evoked`, by `Mohamed A. Sherif`_ in :gh:`478`.

- Changed ``conn_seed`` default to ``None`` (from ``3``) in :func:`~hnn_core.network.add_connection`,
  by `Mattan Pelah`_ in :gh:`492`.

- Add interface to modify attributes of sections in
  :func:`~hnn_core.Cell.modify_section`, by `Nick Tolley`_ in :gh:`481`

 - Add ability to target specific sections when adding drives or connections,
   by `Nick Tolley`_ in :gh:`419`

- Runtime output messages now specify the trial with which each simulation time
  checkpoint belongs too, by `Ryan Thorpe`_ in :gh:`546`.

Bug
~~~
- Fix bugs in drives API to enable: rate constant argument as float; evoked drive with
  connection probability, by `Nick Tolley`_ in :gh:`458`

- Allow regular strings as filenames in :meth:`~hnn_core.Cell_response.write` by
  `Mainak Jas`_ in :gh:`456`.

- Fix to make network output independent of the order in which drives are added to
  the network by making the seed of the random process generating spike times in
  drives use the offset of the gid with respect to the first gid in the population
  by `Mainak Jas`_ in :gh:`462`.

- Negative ``event_seed`` is no longer allowed by `Mainak Jas`_ in :gh:`462`.

- Evoked drive optimization no longer assigns a default timing sigma value to
  a drive if it is not already specified, by `Ryan Thorpe`_ in :gh:`446`.

- Subsets of trials can be indexed when using :func:`~hnn_core.viz.plot_spikes_raster`
  and :func:`~hnn_core.viz.plot_spikes_hist`, by `Nick Tolley`_ in :gh:`472`.

- Add option to plot the averaged dipole in `~hnn_core.viz.plot_dipole` when `dpl`
  is a list of dipoles, by `Huzi Cheng`_ in :gh:`475`.

- Fix bug where :func:`~hnn_core.viz.plot_morphology` did not accurately
  reflect the shape of the cell being simulated, by `Nick Tolley`_ in :gh:`481`

- Fix bug where :func:`~hnn_core.network.pick_connection` did not return an
  empty list when searching non existing connections, by `Nick Tolley`_ in :gh:`515`

- Fix bug in :class:`~hnn_core.MPIBackend` that caused an MPI runtime error
  (``RuntimeError: MPI simulation failed. Return code: 143``), when running a
  simulation with an oversubscribed MPI session on a reduced network, by
  `Ryan Thorpe`_ in :gh:`545`.

- Fix bug where :func:`~hnn_core.network.pick_connection` failed when searching
  for connections with a list of cell types, by `Nick Tolley`_ in :gh:`559`

API
~~~
- Optimization of the evoked drives can be conducted on any :class:`~hnn_core.Network`
  template model by passing a :class:`~hnn_core.Network` instance directly into
  :func:`~hnn_core.optimization.optimize_evoked`. Simulations run during
  optimization can now consist of multiple trials over which the simulated
  dipole is averaged, by `Ryan Thorpe`_ in :gh:`446`.

- `~hnn_core.viz.plot_dipole` now supports separate visualizations of different
  layers, by `Huzi Cheng`_ in :gh:`479`.

.. _0.2:

0.2
---

Notable Changes
---------------
- Local field potentials can now be recorded during simulations
  :ref:`[Example] <sphx_glr_auto_examples_howto_plot_record_extracellular_potentials.py>`

- Ability to optimize parameters to reproduce event related potentials from real data
  :ref:`[Example] <sphx_glr_auto_examples_howto_plot_optimize_evoked.py>`

- Published models using HNN were added and can be loaded via dedicated functions

- Several improvements enabling easy modification of connectivity and cell properties
  :ref:`[Example] <sphx_glr_auto_examples_howto_plot_connectivity.py>`

- Improved visualization including spectral analysis, connectivity, and cell morphology

Changelog
~~~~~~~~~
- Store all connectivity information under :attr:`~hnn_core.Network.connectivity` before building
  the network, by `Nick Tolley`_ in :gh:`276`

- Add new function :func:`~hnn_core.viz.plot_cell_morphology` to visualize cell morphology,
  by `Mainak Jas`_ in :gh:`319`

- Compute dipole component in z-direction automatically from cell morphology instead of hard coding,
  by `Mainak Jas`_ in  :gh:`327`

- Store :class:`~hnn_core.Cell` instances in :class:`~hnn_core.Network`'s :attr:`~/hnn_core.Network.cells`
  attribute by `Ryan Thorpe`_ in :gh:`321`

- Add probability argument to :func:`~hnn_core.Network.add_connection`. Connectivity patterns can also
  be visualized with :func:`~hnn_core.viz.plot_connectivity_matrix`, by `Nick Tolley`_ in :gh:`318`

- Add function to visualize connections originating from individual cells :func:`~hnn_core.viz.plot_cell_connectivity`,
  by `Nick Tolley`_ in :gh:`339`

- Add method for calculating extracellular potentials using electrode arrays
  :func:`~hnn_core.Network.add_electrode_array` that are stored under ``net.rec_array`` as a dictionary
  of :class:`~hnn_core.extracellular.ExtracellularArray` containers, by `Mainak Jas`_,
  `Nick Tolley`_ and `Christopher Bailey`_ in :gh:`329`

- Add function to visualize extracellular potentials from laminar array simulations,
  by `Christopher Bailey`_ in :gh:`329`

- Previously published models can now be loaded via :func:`~hnn_core.law_2021_model()`
  and :func:`~hnn_core.jones_2009_model()`, by `Nick Tolley`_ in :gh:`348`

- Add ability to interactivity explore connections in :func:`~hnn_core.viz.plot_cell_connectivity`
  by `Mainak Jas`_ in :gh:`376`

- Add :func:`~hnn_core.calcium_model` with a distance dependent calcium channel conductivity,
  by `Nick Tolley`_ and `Sarah Pugliese`_ in :gh:`348`

- Each drive spike train sampled through an independent process corresponds to a single artificial
  drive cell, the number of which users can set when adding drives with ``n_drive_cells`` and
  ``cell_specific``, by `Ryan Thorpe`_ in :gh:`383`

- Add :func:`~hnn_core.pick_connection` to query the indices of specific connections in
  :attr:`~hnn_core.Network.connectivity`, by `Nick Tolley`_ in :gh:`367`

- Drives in :attr:`~hnn_core.Network.external_drives` no longer contain a `'conn'` key and the
  :attr:`~hnn_core.Network.connectivity` list contains more items when adding drives from a param
  file or when in legacy mode, by `Ryan Thorpe`_, `Mainak Jas`_, and `Nick Tolley`_ in :gh:`369`

- Add :func:`~hnn_core.optimization.optimize_evoked` to optimize the timing and weights of driving
  inputs for simulating evoked responses, by `Blake Caldwell`_ and `Mainak Jas`_ in :gh:`77`

- Add method for setting in-plane cell distances and layer separation in the network :func:`~hnn_core.Network.set_cell_positions`, by `Christopher Bailey`_ in `#370 <https://github.com/jonescompneurolab/hnn-core/pull/370>`_

- External drives API now accepts probability argument for targetting subsets of cells,
  by `Nick Tolley`_ in :gh:`416`

Bug
~~~

- Remove rounding error caused by repositioning of NEURON cell sections, by `Mainak Jas`_
  and `Ryan Thorpe`_ in :gh:`314`

- Fix issue where common drives use the same parameters for all cell types, by `Nick Tolley`_
  in :gh:`350`

- Fix bug where depth of L5 and L2 cells were swapped, by `Christopher Bailey`_ in :gh:`352`

- Fix bug where :func:`~hnn_core.dipole.average_dipoles` failed when there were less than two dipoles in the
  input dipole list, by `Kenneth Loi`_ in :gh:`368`

- Fix bug where :func:`~hnn_core.read_spikes` wasn't returning a :class:`~hnn_core.CellResponse` instance
  with updated spike types, by `Ryan Thorpe`_ in :gh:`382`

- :attr:`Dipole.times` and :attr:`Cell_response.times` now reflect the actual
  integration points instead of the intended times, by `Mainak Jas`_ in :gh:`397`

- Fix overlapping non-cell-specific drive gid assignment over different ranks in `~hnn_core.MPIBackend`, by `Ryan Thorpe`_
  and `Mainak Jas`_ in :gh:`399`

- Allow :func:`~hnn_core.read_dipoles` to read dipole from a file with only two columns
  (``times`` and ``data``), by `Mainak Jas`_ in :gh:`421`

API
~~~
- New API for defining cell-cell connections. Custom connections can be added with
  :func:`~hnn_core.Network.add_connection`, by `Nick Tolley`_ in :gh:`276`

- Remove :class:`~hnn_core.L2Pyr`, :class:`~hnn_core.L5Pyr`, :class:`~hnn_core.L2Basket`,
  and :class:`~hnn_core.L5Basket` classes in favor of instantation through functions and
  a more consistent :class:`~hnn_core.Cell` class by `Mainak Jas`_ in  :gh:`322`

- Remove parameter ``distribution`` in :func:`~hnn_core.Network.add_bursty_drive`.
  The distribution is now Gaussian by default, by `Mainak Jas`_ in :gh:`330`

- New API for accessing and modifying :class:`~hnn_core.Cell` attributes (e.g., synapse and biophysics parameters)
  as cells are now instantiated from template cells specified in a :class:`~hnn_core.Network`
  instance's :attr:`~/hnn_core.Network.cell_types` attribute by `Ryan Thorpe`_ in :gh:`321`

- New API for network creation. The default network is now created with
  ``net = jones_2009_model(params)``, by `Nick Tolley`_ in :gh:`318`

- Replace parameter ``T`` with ``tstop`` in :func:`~hnn_core.Network.add_tonic_bias`
  and :func:`~hnn_core.Cell.create_tonic_bias` to be more consistent with other functions and
  improve readability, by `Kenneth Loi`_ in :gh:`354`

- Deprecated ``postproc`` argument in :func:`~hnn_core.dipole.simulate_dipole`, whereby user should
  explicitly smooth and scale resulting dipoles, by `Christopher Bailey`_ in :gh:`372`

- Number of drive cells and their connectivity can now be specified through the ``n_drive_cells``
  and ``cell_specific`` arguments in ``Network.add_xxx_drive()`` methods, replacing use of ``repeats``
  and ``sync_within_trial``, by `Ryan Thorpe`_ in :gh:`383`

- Simulation end time and integration time have to be specified now with ``tstop`` and ``dt`` in
  :func:`~hnn_core.simulate_dipole`, by `Mainak Jas`_ in :gh:`397`

- :meth:`CellResponse.reset` method is not supported any more, by `Mainak Jas`_ in :gh:`397`

- Target cell types and their connections are created for each drive according to the synaptic weight
  and delay dictionaries assigned in ``Network.add_xxx_drive()``, by `Ryan Thorpe`_ in :gh:`369`

- Cell objects can no longer be accessed from :class:`~hnn_core.Network` as the
  :attr:`~hnn_core.Network.cells` attribute has been removed, by `Ryan Thorpe`_ in :gh:`436`

People who contributed to this release (in alphabetical order):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Alex Rockhill`_
- `Blake Caldwell`_
- `Christopher Bailey`_
- `Dylan Daniels`_
- `Kenneth Loi`_
- `Mainak Jas`_
- `Nick Tolley`_
- `Ryan Thorpe`_
- `Sarah Pugliese`_
- `Stephanie R. Jones`_

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
- `Stephanie R. Jones`_

.. _Alex Rockhill: https://github.com/alexrockhill
.. _Blake Caldwell: https://github.com/blakecaldwell
.. _Christopher Bailey: https://github.com/cjayb
.. _Carmen Kohl: https://github.com/kohl-carmen
.. _Dylan Daniels: https://github.com/dylansdaniels
.. _Huzi Cheng: https://github.com/chenghuzi
.. _Kenneth Loi: https://github.com/kenloi
.. _Mainak Jas: http://jasmainak.github.io/
.. _Mattan Pelah: https://github.com/mjpelah
.. _Mohamed A. Sherif: https://github.com/mohdsherif/
.. _Nick Tolley: https://github.com/ntolley
.. _Ryan Thorpe: https://github.com/rythorpe
.. _Samika Kanekar: https://github.com/samikane
.. _Sarah Pugliese: https://bcs.mit.edu/directory/sarah-pugliese
.. _Stephanie R. Jones: https://github.com/stephanie-r-jones
