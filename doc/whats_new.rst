:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: hnn_core

Current
-------

Changelog
~~~~~~~~~
- Add button to delete a single drive on GUI drive windows, by
  `George Dang`_ in :gh:`890`

- Add minimum spectral frequency widget to GUI for adjusting spectrogram
  frequency axis, by `George Dang`_ in :gh:`894`

- Add method to modify synaptic gains, by `Nick Tolley`_  and `George Dang`_
  in :gh:`897`

- Update GUI to display "L2/3", by `Austin Soplata`_ in :gh:`904`

Bug
~~~
- Fix GUI over-plotting of loaded data where the app stalled and did not plot
  RMSE, by `George Dang`_ in :gh:`869`

- Fix scaling and smoothing of loaded data dipoles to the GUI, by `George Dang`_
  in :gh:`892`

API
~~~
- Add :func:`~hnn_core.CellResponse.spike_times_by_type` to get cell spiking times
  organized by cell type, by `Mainak Jas`_ in :gh:`916`.

.. _0.4:

0.4
---

Changelog
~~~~~~~~~
- Fix bug in :func:`~hnn_core.Network.add_poisson_drive` where an error is
  thrown when passing an int for rate_constant when ``cell_specific=True``,
  by `Dylan Daniels`_ in :gh:`818`

- Fix bug in :func:`~hnn_core.Network.add_poisson_drive` where an error is
  thrown when passing a float for rate_constant when ``cell_specific=False``,
  by `Dylan Daniels`_ in :gh:`814`
  
- Add ability to customize plot colors for each cell section in
  :func:`~hnn_core.Cell.plot_morphology`, by `Nick Tolley`_ in :gh:`646`
  
- Add ability to manually define colors in spike histogram plots,
  by `Nick Tolley`_ in :gh:`640`

- Update minimum supported version of Python to 3.8, by `Ryan Thorpe`_ in
  :gh:`678`.

- Update GUI to use ipywidgets v8.0.0+ API, by `George Dang`_ in
  :gh:`696`.

- Add dependency groups to setup.py and update CI workflows to reference
  dependency groups, by `George Dang`_ in :gh:`703`.

- Add ability to specify number of cells in :class:`~hnn_core.Network`,
  by `Nick Tolley`_ in :gh:`705`
  
- Fixed figure annotation overlap in multiple sub-plots, 
  by `Camilo Diaz`_ in :gh:`741`

- Fix bug in :func:`~hnn_core.network.pick_connection` where connections are
  returned for cases when there should be no valid matches, by `George Dang`_
  in :gh:`739`

- Added check for invalid Axes object in :func:`~hnn_core.viz.plot_cells` 
  function, by `Abdul Samad Siddiqui`_ in :gh:`744`.

- Added kwargs options to `plot_spikes_hist` for adjusting the histogram plots 
  of spiking activity, by `Abdul Samad Siddiqui`_ in :gh:`732`.
  
- Added pre defined plot sets for simulated data, 
  by `Camilo Diaz`_ in :gh:`746`

- Added gui widget to enable/disable synchronous input in simulations, 
  by `Camilo Diaz`_ in :gh:`750`

- Added gui widgets to save simulation as csv and updated the file upload to support csv data,
  by `Camilo Diaz`_ in :gh:`753`

- Added feature to read/write :class:`~hnn_core.Network` configurations to
  json, by `George Dang`_ and `Rajat Partani`_ in :gh:`757`

- Added :class:`~hnn_core.viz.NetworkPlotter` to visualize and animate network simulations,
  by `Nick Tolley`_ in :gh:`649`.

- Added GUI feature to include Tonic input drives in simulations,
  by `Camilo Diaz` :gh:`773`

- :func:`~plot_lfp`, :func:`~plot_dipole`, :func:`~plot_spikes_hist`,
  and :func:`~plot_spikes_raster` now plotted from 0 to tstop. Inputs tmin and tmax are deprecated,
  by `Katharina Duecker`_ in :gh:`769`

- Add function :func:`~hnn_core.params.convert_to_json` to convert legacy param
  and json files to new json format, by `George Dang`_ in :gh:`772`

- Add :class:`~hnn_core.BatchSimulate` for batch simulation capability,
  by `Abdul Samad Siddiqui`_ in :gh:`782`.

- Updated `plot_spikes_raster` logic to include all neurons in network model.
  Removed GUI exclusion from build, by `Abdul Samad Siddiqui`_  in :gh:`754`.

- Added GUI feature to read and modify cell parameters,
  by `Camilo Diaz`_  in :gh:`806`.
  
- Add ability to optimize parameters associated with rhythmic drives,
  by `Carolina Fernandez Pujol`_ in :gh:`673`.

- Added features to :func:`~plot_csd`: to set color of sinks and sources, range of the colormap,
  and interpolation method to smoothen CSD plot,
  by `Katharina Duecker`_ in :gh:`815`

- Cleaned up internal logic in :class:`~hnn_core.CellResponse`,
  by `Nick Tolley`_ in :gh:`647`.

- Changed the configuration/parameter file format support of the GUI. Loading
  of connectivity and drives use a new multi-level json structure that mirrors
  the structure of the Network object. Flat parameter and json configuration
  files are no longer supported by the GUI, by `George Dang`_ in :gh:`837`

- Updated the GUI load drive widget to be able to load tonic biases from a
  network configuration file. `George Dang`_ in :gh:`852`

- Added "No. Drive Cells" input widget to the GUI and changed the "Synchronous
  Input" checkbox to "Cell-Specific" to align with the API `George Dang`_ in :gh:`861`

Bug
~~~
- Fix inconsistent connection mapping from drive gids to cell gids, by
  `Ryan Thorpe`_ in :gh:`642`.

- Objective function called by :func:`~hnn_core.optimization.optimize_evoked`
  now returns a scalar instead of tuple, by `Ryan Thorpe`_ in :gh:`670`.

- Fix GUI plotting bug due to deprecation of matplotlib color cycling method,
  by `George Dang`_ in :gh:`695`.

- Fix loading of drives in the GUI: drives are now overwritten instead of updated,
  by `Mainak Jas`_ in :gh:`795`.

- Use `np.isin()` in place of `np.in1d()` to address numpy deprecation,
  by `Nick Tolley`_ in :gh:`799.

- Fix drive seeding so that event times are unique across multiple trials,
  by `Nick Tolley`_ in :gh:`810`.

- Fix bug in :func:`~hnn_core.network.clear_drives` where network object are not
  accurately updated, by `Nick Tolley`_ in :gh:`812`.

API
~~~
- :func:`~hnn_core.CellResponse.write` and :func:`~hnn_core.Cell_response.read_spikes`
  now support hdf5 format for read/write Cell response object, by
  `Rajat Partani`_ in :gh:`644`

- Connection `'src_gids'` and `'target_gids'` are now stored as set objects
  instead of lists, by `Ryan Thorpe`_ in :gh:`642`.

- :func:`~hnn_core.Dipole.write` and :func:`~hnn_core.Dipole.read_dipoles`
  now support hdf5 format for read/write Dipole object, by
  `Rajat Partani`_ in :gh:`648`
  
- Add ability to optimize parameters associated with evoked drives and plot 
  convergence. User can constrain parameter ranges and specify solver,
  by `Carolina Fernandez Pujol`_ in :gh:`652`

- :func:`network.add_tonic_bias` cell-specific tonic bias can now be 
  provided using the argument amplitude in network.add_tonic_bias`,
  by `Camilo Diaz`_ in :gh:`766`

People who contributed to this release (in alphabetical order):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Huzi Cheng`_
- `Tianqi Cheng`_
- `Dylan Daniels`_
- `George Dang`_
- `Camilo Diaz`_
- `Katharina Duecker`_
- `Carolina Fernandez Pujol`_
- `Yaroslav Halchenko`_
- `Mainak Jas`_
- `Nick Tolley`_
- `Orsolya Beatrix Kolozsvari`_
- `Rajat Partani`_
- `Abdul Samad Siddiqui`_
- `Ryan Thorpe`_
- `Stephanie R. Jones`_

.. _0.3:

0.3
---

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

- Add warning if network drives are not loaded, by `Orsolya Beatrix Kolozsvari`_ in :gh:`516`

- Add ability to record voltages and synaptic currents from all sections in :class:`~hnn_core.CellResponse`,
  by `Nick Tolley`_ in :gh:`502`.

- Add ability to return unweighted RMSE for each optimization iteration in :func:`~hnn_core.optimization.optimize_evoked`, by `Kaisu Lankinen`_ in :gh:`610`.

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

- Fix bug where :func:`~hnn_core.network.add_evoked_drive` failed when adding
  a drive with just NMDA weights, by `Nick Tolley`_ in :gh:`611`

- Fix bug where :func:`~hnn_core.params.read_params` failed to create a network when
  legacy mode is False, by `Nick Tolley`_ in :gh:`614`

- Fix bug where :func:`~hnn_core.viz.plot_dipole` failed to check the instance
  type of Dipole, by `Rajat Partani`_ in :gh:`606`

API
~~~
- Optimization of the evoked drives can be conducted on any :class:`~hnn_core.Network`
  template model by passing a :class:`~hnn_core.Network` instance directly into
  :func:`~hnn_core.optimization.optimize_evoked`. Simulations run during
  optimization can now consist of multiple trials over which the simulated
  dipole is averaged, by `Ryan Thorpe`_ in :gh:`446`.

- `~hnn_core.viz.plot_dipole` now supports separate visualizations of different
  layers, by `Huzi Cheng`_ in :gh:`479`.

- Current source density (CSD) can now be calculated with
  :func:`~hnn_core.extracellular.calculate_csd2d` and plotted with
  :meth:`~hnn_core.extracellular.ExtracellularArray.plot_csd`. The method for
  plotting local field potential (LFP) is now found at
  :meth:`~hnn_core.extracellular.ExtracellularArray.plot_lfp`, by
  `Steven Brandt`_ and `Ryan Thorpe`_ in :gh:`517`.

- Recorded voltages/currents from the soma, as well as all sections, are enabled by
  setting either `record_vsec` or `record_isec` to `'all'` or `'soma'` 
  in :func:`~hnn_core.simulate_dipole`. Recordings are now accessed through
  :class:`~hnn_core.CellResponse.vsec` and :class:`~hnn_core.CellResponse.isec`,
  by `Nick Tolley`_ in :gh:`502`.

- legacy_mode is now set to False by default in all for all
  :class:`~hnn_core.Network` objects, 
  by `Nick Tolley`_ and `Ryan Thorpe`_ in :gh:`619`.

- Recorded calcium conncetration from the soma, as well as all sections, are enabled
  by setting `record_ca` to `soma` or `all` in :func:`~hnn_core.simulate_dipole`.
  Recordings are accessed through :class:`~hnn_core.CellResponse.ca`, 
  by `Katharina Duecker`_ in :gh:`804`

People who contributed to this release (in alphabetical order):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Christopher Bailey`_
- `Huzi Cheng`_
- `Kaisu Lankinen`_
- `Mainak Jas`_
- `Mattan Pelah`_
- `Mohamed A. Sherif`_
- `Mostafa Khalil`_
- `Nick Tolley`_
- `Orsolya Beatrix Kolozsvari`_
- `Rajat Partani`_
- `Ryan Thorpe`_
- `Stephanie R. Jones`_
- `Steven Brandt`_

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

- External drives API now accepts probability argument for targeting subsets of cells,
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
  and :class:`~hnn_core.L5Basket` classes in favor of instantiation through functions and
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
.. _Mostafa Khalil: https://github.com/mkhalil8
.. _Nick Tolley: https://github.com/ntolley
.. _Orsolya Beatrix Kolozsvari: http://github.com/orbekolo/
.. _Rajat Partani: https://github.com/raj1701
.. _Ryan Thorpe: https://github.com/rythorpe
.. _Samika Kanekar: https://github.com/samikane
.. _Sarah Pugliese: https://bcs.mit.edu/directory/sarah-pugliese
.. _Stephanie R. Jones: https://github.com/stephanie-r-jones
.. _Steven Brandt: https://github.com/spbrandt
.. _Kaisu Lankinen: https://github.com/klankinen
.. _George Dang: https://github.com/gtdang
.. _Camilo Diaz: https://github.com/kmilo9999
.. _Abdul Samad Siddiqui: https://github.com/samadpls
.. _Katharina Duecker: https://github.com/katduecker
.. _Yaroslav Halchenko:  https://github.com/yarikoptic
.. _Tianqi Cheng: https://github.com/tianqi-cheng
