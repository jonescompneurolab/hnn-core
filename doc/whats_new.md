---
orphan: true
---

(whats_new)=
# What's new?

## Current development version

### API Changes

- :class:`~hnn_core.CellResponse` now requires a `cell_type_names` argument, whereas before the argument was optional.

### Changelog

- Remove hardcoding of celltypes in :class:`~hnn_core.CellResponse` and add
  {func}`~hnn_core.Network.rename_cell`, by [Mohamed W. ElSayed][] in {gh}`702` and {gh}`970`.

## 0.4.1 Patch notes

- Version 0.4.1 is a bug-fixing patch release for version 0.4. This includes changes to importing of `BatchSimulate` due to previously-undetected install/import issues ({gh}`1034`), configuration of packaging metadata format (same PR), and elimination of a discrepancy in our method of cleaning local compiled files that led to architecture-specific files being included in the Pypi 0.4 release, which caused simulations on some platforms to fail ({gh}`1035`). The public Pypi version has already been updated to 0.4.1.

## 0.4 Release Notes

v0.4 represents a major milestone in development of `hnn_core` and the HNN ecosystem as a whole. v0.4 includes over *two years* of active development work by many people (>800 commits!), and brings with it many new and exciting features, including significant improvements to robustness, testing, and bug-fixing.

### New Features

- `hnn_core` now includes a fully-tested and robust GUI of its own. The `hnn_core` GUI was present as a prototype in v0.3, but it is now ready for production. New features and visual improvements will still be coming to it in the future, such as the ability to use optimization. See our new [Install page](https://jonescompneurolab.github.io/hnn-core/dev/install.html) for ways to install it, and we have already begun incorporating it into a new, fresh series of tutorials for our upcoming revamp of the HNN website. If you have installed it, you can start the GUI using `hnn-gui` in your terminal/command prompt window.

- The `BatchSimulate` class: Thanks to [Abdul Samad Siddiqui][] and Google Summer of Code 2024, there is now the capability to run "batches" of simulations across multiple parameter sets, enabling easy analysis and simulation of behavior across parameter sweeps. See our [example for more details](https://jonescompneurolab.github.io/hnn-core/dev/auto_examples/howto/plot_batch_simulate.html#sphx-glr-auto-examples-howto-plot-batch-simulate-py). Note that currently, only its `loky` backend is supported, and the `"hnn-core[parallel]"` dependencies must be installed for it to be used.

- Significant improvements to the API, documentation, and pedagogical examples [especially for Optimization](https://jonescompneurolab.github.io/hnn-core/stable/auto_examples/howto/optimize_evoked.html#sphx-glr-auto-examples-howto-optimize-evoked-py), among others.

- Calcium concentration can now be recorded: recorded calcium concentration from either the soma,
  or all sections, are enabled by setting `record_ca` to `soma` or `all` in
  {func}`~hnn_core.simulate_dipole`. Recordings are accessed through
  {class}`~hnn_core.CellResponse.ca`.

- There is now a new class {class}`~hnn_core.viz.NetworkPlotter` which can be used to visualize an entire network in 3D, including firing animations; [see our example of how to use it here](https://jonescompneurolab.github.io/hnn-core/dev/auto_examples/howto/plot_hnn_animation.html#sphx-glr-auto-examples-howto-plot-hnn-animation-py).

- There is now a new function {func}`~hnn_core.viz.plot_drive_strength` for illustrating the absolute or relative amount of strength that a particular drive provides to different cell types.

- A very large amount of polishing, bug fixes, general improvements, etc.

### Deprecations

- The new Python 3.13 is **not** supported by `hnn_core` at this time, due to [NEURON](https://nrn.readthedocs.io/en/8.2.6/)'s current lack of support for it. This will change in the near future. We still support 3.8 through 3.12 (inclusively).

### Upcoming Deprecations

- Both {func}`~hnn_core.viz.plot_laminar_lfp` and {func}`~hnn_core.viz.plot_dipole` will have their `tmin` and `tmax` arguments removed in the future. Please set the x-axis limits using methods called directly on the existing `matplotlib` objects, or using [`matplotlib.pyplot.xlim`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlim.html#matplotlib.pyplot.xlim).
- {class}`~hnn_core.Network`'s argument of `legacy_mode` for importing old param files will be removed in the future.
- {func}`~hnn_core.Network.add_tonic_bias`'s argument of `cell_type`, along with setting the argument `amplitude` to a single float, will be removed in the future. Instead, set the `amplitude` argument to a dictionary as described in the docstring.
- {func}`~hnn_core.simulate_dipole`'s argument of `postproc` for post-processing will be removed in the future. Instead, use explicit smoothing and scaling via {class}`~hnn_core.Dipole` methods.

### API Changes

- New argument to {class}`~hnn_core.Network` initialization: you can now set `mesh_shape` to easily make a grid of different sizes of `Network`s.
- {class}`~hnn_core.Cell` initialization argument `topology` has had both its name changed to `cell_tree` and its data type significantly changed; see [the API docs of `Cell` for details](https://jonescompneurolab.github.io/hnn-core/dev/generated/hnn_core.Cell.html#hnn_core.Cell).
- {func}`~hnn_core.jones_2009_model` and other built-in Network Models including {func}`~hnn_core.law_2021_model` and {func}`~hnn_core.calcium_model` all accept the aforementioned `mesh_shape` argument like {class}`~hnn_core.Network`.
- The API for optimization has changed significantly. Instead of running the function `optimize_evoked` obtained using `from hnn_core.optimization import optimize_evoked`, you should use the new {class}`~hnn_core.Optimizer` class and its methods; [see our example of evoked-response optimization here](https://jonescompneurolab.github.io/hnn-core/dev/auto_examples/howto/optimize_evoked.html#sphx-glr-auto-examples-howto-optimize-evoked-py).
- {func}`~hnn_core.viz.plot_spikes_hist` now accepts more arguments, including `invert_spike_types`, `color`, and any `**kwargs_hist` which can be applied to `matplotlib.axes.Axes.hist`. See the docstring for details.
- {func}`~hnn_core.viz.plot_spikes_raster` now accepts many more arguments, including `cell_types`, `colors`, `show_legend`, `marker_size`, `dpl`, and `overlay_dipoles`. See the docstring for details.
- {func}`~hnn_core.viz.plot_cell_morphology` now accepts more arguments, including `color` and several arguments related to its position and viewing window, including `pos`, `xlim`, `ylim`, and `zlim`. See the docstring for details.
- {func}`~hnn_core.viz.plot_laminar_csd` now accepts more arguments, including `vmin`, `vmax`, `sink`, and `interpolation`. See the docstring for details.
- {class}`~hnn_core.parallel_backends.MPIBackend` now accepts many more arguments, including `use_hwthreading_if_found`, `sensible_default_cores`, `override_hwthreading_option`, and `override_oversubscribe_option`. See the docstring for details; the ability to customize it has been greatly increased.
- {func}`~hnn_core.read_params` now accepts a new argument `file_contents` which lets you pass in network configuration contents using a string.

### People who contributed to this release (in alphabetical order of family name):

- [Huzi Cheng][]
- [Tianqi Cheng][]
- [George Dang][]
- [Dylan Daniels][]
- [Camilo Diaz][]
- [Katharina Duecker][]
- [Yaroslav Halchenko][]
- [Mainak Jas][]
- [Dikshant Jha][]
- [Stephanie R. Jones][]
- [Shehroz Kashif][]
- [Rajat Partani][]
- [Carolina Fernandez Pujol][]
- [Dan Toms][]
- [Abdul Samad Siddiqui][]
- [Austin E. Soplata][]
- [Ryan Thorpe][]
- [Nick Tolley][]

### PRs merged (API)

- Add ability to manually define colors in spike histogram plots, by [Nick Tolley][] in
  {gh}`640`

- Connection `'src_gids'` and `'target_gids'` are now stored as set objects instead of
  lists, by [Ryan Thorpe][] in {gh}`642`

- {func}`~hnn_core.CellResponse.write` and {func}`~hnn_core.Cell_response.read_spikes`
  now support hdf5 format for read/write Cell response object, by [Rajat Partani][] in
  {gh}`644` (Note: this work was later reverted in {gh}`654`)

- Add ability to customize plot colors for each cell section in
  {func}`~hnn_core.Cell.plot_morphology`, by [Nick Tolley][] in {gh}`646`

- {func}`~hnn_core.Dipole.write` and {func}`~hnn_core.Dipole.read_dipoles` now support
  hdf5 format for read/write Dipole object, by [Rajat Partani][] in {gh}`648`

- Added {class}`~hnn_core.viz.NetworkPlotter` to visualize and animate network
  simulations, by [Nick Tolley][] in {gh}`649`

- Add ability to optimize parameters associated with evoked drives and plot
  convergence. User can constrain parameter ranges and specify solver, by [Carolina
  Fernandez Pujol][] in {gh}`652`

- {func}`~hnn_core.CellResponse` no longer supports reading and writing to hdf5, by
  [Rajat Partani][] and [Nick Tolley][], in {gh}`654`

- Add ability to optimize parameters associated with rhythmic drives, by [Carolina
  Fernandez Pujol][] in {gh}`673`

- Add initial support for {class}`~hnn_core.Network` read and write of hdf5 files by
  [George Dang][] in {gh}`704` (Note: this work was later obviated by {gh}`756`)

- Add ability to specify number of cells in {class}`~hnn_core.Network`, by [Nick
  Tolley][] in {gh}`705`

- Added kwargs options to `plot_spikes_hist` for adjusting the histogram plots of
  spiking activity, by [Abdul Samad Siddiqui][] in {gh}`732`

- Updated `plot_spikes_raster` logic to include all neurons in network model.  Removed
  GUI exclusion from build, by [Abdul Samad Siddiqui][] in {gh}`754`

- Add initial work on hierarchical json format in place of hdf5, by [George Dang][] in
  {gh}`763`

- {func}`network.add_tonic_bias` cell-specific tonic bias can now be provided using the
  argument amplitude in {func}`network.add_tonic_bias`, by [Camilo Diaz][] in {gh}`766`

- {func}`~plot_lfp`, {func}`~plot_dipole`, {func}`~plot_spikes_hist`, and
  {func}`~plot_spikes_raster` now plotted from 0 to tstop. Inputs tmin and tmax are
  deprecated, by [Katharina Duecker][] in {gh}`769`

- Add function {func}`~hnn_core.params.convert_to_json` to convert legacy param and json
  files to new json format, by [George Dang][] in {gh}`772`

- Add
  [`BatchSimulate`](https://jonescompneurolab.github.io/hnn-core/dev/auto_examples/howto/plot_batch_simulate.html#sphx-glr-auto-examples-howto-plot-batch-simulate-py)
  class for batch simulation capability, by [Abdul Samad Siddiqui][]
  in {gh}`782`

- Recorded calcium concentration from the soma, as well as all sections, are enabled by
  setting `record_ca` to `soma` or `all` in {func}`~hnn_core.simulate_dipole`.
  Recordings are accessed through {class}`~hnn_core.CellResponse.ca`, by [Katharina
  Duecker][] in {gh}`804`

- Added features to {func}`~plot_csd`: to set color of sinks and sources, range of the
  colormap, and interpolation method to smoothen CSD plot, by [Katharina Duecker][] in
  {gh}`815`

- Refactor and improve documentation for
  [`BatchSimulate`](https://jonescompneurolab.github.io/hnn-core/dev/auto_examples/howto/plot_batch_simulate.html#sphx-glr-auto-examples-howto-plot-batch-simulate-py), by [Abdul Samad Siddiqui][]
  in {gh}`830` and {gh}`857`

- Add argument to change colors of `plot_spikes_raster`, shortened line lengths to
  prevent overlap, and added an argument for custom cell types, by [George Dang][] in
  {gh}`895`

- Add method to {class}`~hnn_core.Network` to modify synaptic gains, by [Nick Tolley][]
  and [George Dang][] in {gh}`897`

- Add {func}`~hnn_core.CellResponse.spike_times_by_type` to get cell spiking times
  organized by cell type, by [Mainak Jas][] in {gh}`916`

- Add option to apply a tonic bias to any compartment of the cell, and option to add
  multiple biases per simulation and cell {func}`hnn_core.network.add_tonic_bias`, by
  [Katharina Duecker][] in {gh}`922`

- Add plots to show relative and absolute external drive strength, by [Dikshant Jha][]
  in {gh}`987`

- Make re-generation of our testing network usable and explicit, by [Austin E. Soplata][]
  in {gh}`988`

- Improvements to raster plotting, by [Dylan Daniels][] in
  {gh}`1017` and {gh}`1018`

### PRs merged (Bug fixes and corrections)

- Objective function called by {func}`~hnn_core.optimization.optimize_evoked` now
  returns a scalar instead of tuple, by [Ryan Thorpe][] in {gh}`670`

- Fix error message for drive addition, by [Tianqi Cheng][] in {gh}`681`

- Fix GUI plotting bug due to deprecation of matplotlib color cycling method, by [George
  Dang][] in {gh}`695`

- Typo fix, by [George Dang][] in {gh}`707`

- Fix GUI dipole plot scale and smooth factors, by [Camilo Diaz][] in {gh}`730`

- Fix file upload widget, by [Camilo Diaz][] in {gh}`736`

- Fix bug in {func}`~hnn_core.network.pick_connection` where connections are returned
  for cases when there should be no valid matches, by [George Dang][] in {gh}`739`

- Fix GUI figure annotation overlap in multiple sub-plots, by [Camilo Diaz][] in
  {gh}`741`

- Various CI updates and fixes, by [George Dang][] in {gh}`758`

- Fix GUI load data button size, by [Camilo Diaz][] in {gh}`775`

- Fix typos, by [George Dang][] in {gh}`777`

- Fix CI Linux conda bug, by [Camilo Diaz][] in {gh}`794`

- Fix loading of drives in the GUI: drives are now overwritten instead of updated, by
  [Mainak Jas][] in {gh}`795`

- Use `np.isin()` in place of `np.in1d()` to address numpy deprecation, by [Nick
  Tolley][] in {gh}`799`

- Fix unit tests for Python 3.11 and 3.12, by [Camilo Diaz][] in {gh}`800`

- Fix README badge URL, by [Camilo Diaz][] in {gh}`802`

- Fix NEURON download link, by [Camilo Diaz][] in {gh}`803`

- Fix clearance of drive connections by network config read, by [George Dang][] in
  {gh}`807`

- Fixes for Binder notebook usage, by [Mainak Jas][] in {gh}`809`, {gh}`820`, and
  {gh}`822`

- Fix drive seeding so that event times are unique across multiple trials, by [Nick
  Tolley][] in {gh}`810`

- Fix bug in {func}`~hnn_core.network.clear_drives` where network object are not
  accurately updated, by [Nick Tolley][] in {gh}`812`

- Fix bug in {func}`~hnn_core.Network.add_poisson_drive` where an error is thrown when
  passing a float for rate_constant when ``cell_specific=False``, by [Dylan Daniels][]
  in {gh}`814`

- Fix bug in {func}`~hnn_core.Network.add_poisson_drive` where an error is thrown when
  passing an int for rate_constant when ``cell_specific=True``, by [Dylan Daniels][] in
  {gh}`818`

- Fix homepage links, by [Dylan Daniels][] in {gh}`819`

- Fix GUI simulations dropdown, by [Camilo Diaz][] in {gh}`825` and {gh}`827`

- Fix argument pass during conversion of network config file to
  {class}`~hnn_core.Network`, by [George Dang][] in {gh}`834`

- Fix GUI visualization, by [Camilo Diaz][] in {gh}`836`

- Fix GUI probability assignment, by [George Dang][] in {gh}`844`

- Fix GUI unnecessary display call, by [George Dang][] in {gh}`845`

- Fix GUI drive sorting, by [George Dang][] in {gh}`851`

- Fix persistent linkcheck failure, by [George Dang][] in {gh}`854`

- Fix GUI MPI test, by [George Dang][] in {gh}`868`

- Fix GUI over-plotting of loaded data where the app stalled and did not plot RMSE, by
  [George Dang][] in {gh}`869`

- Fix GUI MPI cores, by [George Dang][] in {gh}`871`

- Fix GUI output log, by [George Dang][] in {gh}`873`

- Fix MPIBackend platform logic, by [George Dang][] in {gh}`876`

- Fix scaling and smoothing of loaded data dipoles to the GUI, by [George Dang][] in
  {gh}`892`

- Fix minor GUI glitches, by [George Dang][] in {gh}`899`

- Fix GUI synapses properties rendering, by [George Dang][] in {gh}`913`

- Fix accidental removal of second axis object in GUI by [Austin E. Soplata][] in
  {gh}`929`

- Fix statistical Poisson drive tests that were failing stochastically by [Austin
  E. Soplata][] in {gh}`978`

- Fix typo of "leading" in docstring, by [Dan Toms][] in {gh}`979`

- Copy template for monthly metrics workflow, in the hope it will fix the unsuccessful
  runs by [Austin E. Soplata][] in {gh}`983`

- Hotfix of MPI install on MacOS CI runners by [Austin E. Soplata][] in {gh}`994`

- Fix MPI test failures probably due to incomplete Network destruction by [Austin E. Soplata][] in
  {gh}`1010`

### PRs merged (GUI changes)

- Add RMSE calculation and plotting to GUI, by [Huzi Cheng][] in {gh}`636`

- Update GUI to use ipywidgets v8.0.0+ API, by [George Dang][] in {gh}`696`

- Add GUI visualization testing, by [Abdul Samad Siddiqui][] in {gh}`726`

- Add pre defined plot sets for simulated data in GUI, by [Camilo Diaz][] in {gh}`746`

- Add GUI widget to enable/disable synchronous input in simulations, by [Camilo Diaz][]
  in {gh}`750`

- Add GUI widgets to save simulation as csv and updated the file upload to support csv
  data, by [Camilo Diaz][] in {gh}`753`

- Refactor GUI tests, by [George Dang][] in {gh}`765`

- Refactor GUI import of `_read_dipole_text` function, by [George Dang][] in {gh}`771`

- Add GUI feature to include Tonic input drives in simulations, by [Camilo Diaz][]
  {gh}`773`

- Add GUI feature to read and modify cell parameters, by [Camilo Diaz][] in {gh}`806`

- Refactor GUI `read_network_configuration`, by [George Dang][] in {gh}`833`

- Change the configuration/parameter file format support of the GUI. Loading of
  connectivity and drives use a new multi-level json structure that mirrors the
  structure of the Network object. Flat parameter and json configuration files are no
  longer supported by the GUI, by [George Dang][] in {gh}`837`

- GUI load confirmation message, by [George Dang][] in {gh}`846`

- Differentiate L5/L2 Pyr geomtetry options in GUI, by [Nick Tolley][] in {gh}`848`

- Updated the GUI load drive widget to be able to load tonic biases from a network
  configuration file. [George Dang][] in {gh}`852`

- Update GUI initialization of network, by [George Dang][] in {gh}`853`

- Update GUI color, by [Nick Tolley][] in {gh}`855`

- Added "No. Drive Cells" input widget to the GUI and changed the "Synchronous Input"
  checkbox to "Cell-Specific" to align with the API [George Dang][] in {gh}`861`

- Add GUI export of configurations, [George Dang][] in {gh}`862`

- Add screenshot of GUI to README, [George Dang][] in {gh}`865` and {gh}`866`

- Add button to delete a single drive on GUI drive windows, by [George Dang][] in
  {gh}`890`

- Add post-processing for GUI figures, by [George Dang][] in {gh}`893`

- Add minimum spectral frequency widget to GUI for adjusting spectrogram frequency axis,
  by [George Dang][] in {gh}`894`

- Update GUI to display "L2/3", by [Austin E. Soplata][] in {gh}`904`

- Update PSD plot in GUI to use plot config provided frequencies instead of hard-coded
  values, by [Dylan Daniels][] in {gh}`914`

- Flip drives in input histogram based on position in GUI by [Dylan Daniels][] in
  {gh}`923`

- Add GUI widget to adjust default smoothing value, by [Dylan Daniels][] in {gh}`924`

- Change Morlet cycles divisor for better alpha spectral plotting by [Austin
  E. Soplata][] in {gh}`928`

- Add GUI log error message if spectral arguments are invalid by [Austin E. Soplata][]
  in {gh}`944`

- Move GUI log messages to bottom of output by [George Dang][] in {gh}`946`

- Add GUI frequency default visualization parameters and many other smaller visual
  changes by [Dylan Daniels][] in {gh}`952`

- Capture printed messages to logger in GUI by [Dylan Daniels][] in {gh}`956`

- Correctly set layer-specific dipole axes limits in GUI, by [Dylan Daniels][] in
  {gh}`1022`

- Improve spike raster plot by overlaying dipoles, by [Dylan Daniels][] in
  {gh}`1026`

### PRs merged (Other)

- Add Github Discussions installation template, by [Mainak Jas][] in {gh}`630`

- Replace NEURON functions like `define_shape()` and `distance()` with Python
  equivalent, by [Rajat Partani][] in {gh}`661`

- Cleaned up internal logic in {class}`~hnn_core.CellResponse`, by [Nick Tolley][] in
  {gh}`647`

- Add section for JOSS to Readme, by [Ryan Thorpe][] in {gh}`677`

- Update minimum supported version of Python to 3.8, by [Ryan Thorpe][] in {gh}`678`

- Add support for `codespell` checking, by [Yaroslav Halchenko][] in {gh}`692`

- Add citation info to repository, by [Ryan Thorpe][] in {gh}`700`

- Add dependency groups to setup.py and update CI workflows to reference dependency
  groups, by [George Dang][] in {gh}`703`

- Rename io to hnn_io, by [George Dang][] in {gh}`727`

- Expand gitignore to virtual environment directories, by [Abdul Samad Siddiqui][] in
  {gh}`740`

- Add check for invalid Axes object in {func}`~hnn_core.viz.plot_cells` function, by
  [Abdul Samad Siddiqui][] in {gh}`744`

- Refactor pick connection tests, by [George Dang][] in {gh}`745`

- Add governance structure and similar changes, by [Dylan Daniels][] in {gh}`785`

- Add issue metrics Github Action {gh}`790` and associated cron job {gh}`793`, by [Nick
  Tolley][]

- Remove nbsphinx and pandoc usage, by [Nick Tolley][] in {gh}`813`

- Remove nulled drives during convert to hierarchical json function, by [George Dang][]
  in {gh}`821`

- Speedup optimization tests, by [Nick Tolley][] in {gh}`839`

- Add GSoC 2024 acknowledgement, by [Abdul Samad Siddiqui][] in {gh}`874`

- Remove deprecated `distutils` import, by [George Dang][] in {gh}`880`

- Refactor {class}`~hnn_core.Network`'s `__eq__` equivalency function, by [George
  Dang][] in {gh}`902`

- Add automatic spectrogram frequency range reversal, by [Abdul Samad Siddiqui][] in
  {gh}`903`

- Flip drives in input histogram based on position, by [Dylan Daniels][] in {gh}`905`

- Change default smoothing for dipoles to be 0 (only in GUI), by [George Dang][] in
  {gh}`920`

- Add support for parallelizing tests by [Austin E. Soplata][] in {gh}`932`

- Replace `flake8` linting with `ruff check` linting by [Austin E. Soplata][] in
  {gh}`961`

- Add `Makefile` cleanup of arm64-generated files by [Austin E. Soplata][] in {gh}`964`

- Change Sphinx theme, fixing javascript bugs with code-website, and fix some small
  typos by [Austin E. Soplata][] in {gh}`971`

- Replace most ReStructured Text of code-website with Markdown by [Austin E. Soplata][]
  in {gh}`973`

- Add install and run of `codespell` to local testing by [Austin E. Soplata][] in
  {gh}`977`

- Separate Installation to its own page, also other small authoring changes by [Austin E. Soplata][]
  in {gh}`980`

- Update Sphinx `versions.json` link to point to `dev` version, by [Austin E. Soplata][]
  in {gh}`991`

- Add docstring to `_add_cell_type_bias` by [Shehroz Kashif][] in {gh}`1001`

## 0.3

### Changelog

- Add option to select drives using argument 'which_drives' in
  {func}`~hnn_core.optimization.optimize_evoked`, by [Mohamed A. Sherif][] in {gh}`478`

- Changed ``conn_seed`` default to ``None`` (from ``3``) in {func}`~hnn_core.network.add_connection`,
  by [Mattan Pelah][] in {gh}`492`.

- Add interface to modify attributes of sections in
  {func}`~hnn_core.Cell.modify_section`, by [Nick Tolley][] in {gh}`481`

- Add ability to target specific sections when adding drives or connections,
  by [Nick Tolley][] in {gh}`419`

- Runtime output messages now specify the trial with which each simulation time
  checkpoint belongs too, by [Ryan Thorpe][] in {gh}`546`.

- Add warning if network drives are not loaded, by [Orsolya Beatrix Kolozsvari][] in {gh}`516`

- Add ability to record voltages and synaptic currents from all sections in {class}`~hnn_core.CellResponse`,
  by [Nick Tolley][] in {gh}`502`.

- Add ability to return unweighted RMSE for each optimization iteration in {func}`~hnn_core.optimization.optimize_evoked`, by [Kaisu Lankinen][] in {gh}`610`.

### Bug

- Fix bugs in drives API to enable: rate constant argument as float; evoked drive with
  connection probability, by [Nick Tolley][] in {gh}`458`

- Allow regular strings as filenames in {meth}`~hnn_core.Cell_response.write` by
  [Mainak Jas][] in {gh}`456`.

- Fix to make network output independent of the order in which drives are added to
  the network by making the seed of the random process generating spike times in
  drives use the offset of the gid with respect to the first gid in the population
  by [Mainak Jas][] in {gh}`462`.

- Negative ``event_seed`` is no longer allowed by [Mainak Jas][] in {gh}`462`.

- Evoked drive optimization no longer assigns a default timing sigma value to
  a drive if it is not already specified, by [Ryan Thorpe][] in {gh}`446`.

- Subsets of trials can be indexed when using {func}`~hnn_core.viz.plot_spikes_raster`
  and {func}`~hnn_core.viz.plot_spikes_hist`, by [Nick Tolley][] in {gh}`472`.

- Add option to plot the averaged dipole in {func}`~hnn_core.viz.plot_dipole` when `dpl`
  is a list of dipoles, by [Huzi Cheng][] in {gh}`475`.

- Fix bug where {func}`~hnn_core.viz.plot_morphology` did not accurately
  reflect the shape of the cell being simulated, by [Nick Tolley][] in {gh}`481`

- Fix bug where {func}`~hnn_core.network.pick_connection` did not return an
  empty list when searching non existing connections, by [Nick Tolley][] in {gh}`515`

- Fix bug in {class}`~hnn_core.MPIBackend` that caused an MPI runtime error
  (``RuntimeError: MPI simulation failed. Return code: 143``), when running a
  simulation with an oversubscribed MPI session on a reduced network, by
  [Ryan Thorpe][] in {gh}`545`.

- Fix bug where {func}`~hnn_core.network.pick_connection` failed when searching
  for connections with a list of cell types, by [Nick Tolley][] in {gh}`559`

- Fix bug where {func}`~hnn_core.network.add_evoked_drive` failed when adding
  a drive with just NMDA weights, by [Nick Tolley][] in {gh}`611`

- Fix bug where {func}`~hnn_core.params.read_params` failed to create a network when
  legacy mode is False, by [Nick Tolley][] in {gh}`614`

- Fix bug where {func}`~hnn_core.viz.plot_dipole` failed to check the instance
  type of Dipole, by [Rajat Partani][] in {gh}`606`

### API

- Optimization of the evoked drives can be conducted on any {class}`~hnn_core.Network`
  template model by passing a {class}`~hnn_core.Network` instance directly into
  {func}`~hnn_core.optimization.optimize_evoked`. Simulations run during
  optimization can now consist of multiple trials over which the simulated
  dipole is averaged, by [Ryan Thorpe][] in {gh}`446`.

- {func}`~hnn_core.viz.plot_dipole` now supports separate visualizations of different layers, by [Huzi
  Cheng][] in {gh}`479`.

- Current source density (CSD) can now be calculated with
  {func}`~hnn_core.extracellular.calculate_csd2d` and plotted with
  {meth}`~hnn_core.extracellular.ExtracellularArray.plot_csd`. The method for plotting local field
  potential (LFP) is now found at {meth}`~hnn_core.extracellular.ExtracellularArray.plot_lfp`, by
  [Steven Brandt][] and [Ryan Thorpe][] in {gh}`517`.

- Recorded voltages/currents from the soma, as well as all sections, are enabled by setting either
  `record_vsec` or `record_isec` to `'all'` or `'soma'` in
  {func}`~hnn_core.simulate_dipole`. Recordings are now accessed through
  {class}`~hnn_core.CellResponse.vsec` and {class}`~hnn_core.CellResponse.isec`, by [Nick Tolley][]
  in {gh}`502`.

- legacy_mode is now set to False by default in all for all {class}`~hnn_core.Network` objects, by
  [Nick Tolley][] and [Ryan Thorpe][] in {gh}`619`.

### People who contributed to this release (in alphabetical order):

- [Christopher Bailey][]
- [Huzi Cheng][]
- [Kaisu Lankinen][]
- [Mainak Jas][]
- [Mattan Pelah][]
- [Mohamed A. Sherif][]
- [Mostafa Khalil][]
- [Nick Tolley][]
- [Orsolya Beatrix Kolozsvari][]
- [Rajat Partani][]
- [Ryan Thorpe][]
- [Stephanie R. Jones][]
- [Steven Brandt][]

## 0.2

## Notable Changes

- Local field potentials can now be recorded during simulations {ref}`[Example]
  <sphx_glr_auto_examples_howto_plot_record_extracellular_potentials.py>`

- Ability to optimize parameters to reproduce event related potentials from real data
  {ref}`[Example] <sphx_glr_auto_examples_howto_optimize_evoked.py>`

- Published models using HNN were added and can be loaded via dedicated functions

- Several improvements enabling easy modification of connectivity and cell properties
  {ref}`[Example] <sphx_glr_auto_examples_howto_plot_connectivity.py>`

- Improved visualization including spectral analysis, connectivity, and cell morphology

### Changelog

- Store all connectivity information under {attr}`~hnn_core.Network.connectivity` before building
  the network, by [Nick Tolley][] in {gh}`276`

- Add new function {func}`~hnn_core.viz.plot_cell_morphology` to visualize cell morphology, by
  [Mainak Jas][] in {gh}`319`

- Compute dipole component in z-direction automatically from cell morphology instead of hard coding,
  by [Mainak Jas][] in {gh}`327`

- Store {class}`~hnn_core.Cell` instances in {class}`~hnn_core.Network`'s
  {attr}`~/hnn_core.Network.cells` attribute by [Ryan Thorpe][] in {gh}`321`

- Add probability argument to {func}`~hnn_core.Network.add_connection`. Connectivity patterns can
  also be visualized with {func}`~hnn_core.viz.plot_connectivity_matrix`, by [Nick Tolley][] in
  {gh}`318`

- Add function to visualize connections originating from individual cells
  {func}`~hnn_core.viz.plot_cell_connectivity`, by [Nick Tolley][] in {gh}`339`

- Add method for calculating extracellular potentials using electrode arrays
  {func}`~hnn_core.Network.add_electrode_array` that are stored under ``net.rec_array`` as a
  dictionary of {class}`~hnn_core.extracellular.ExtracellularArray` containers, by [Mainak Jas][],
  [Nick Tolley][] and [Christopher Bailey][] in {gh}`329`

- Add function to visualize extracellular potentials from laminar array simulations, by [Christopher
  Bailey][] in {gh}`329`

- Previously published models can now be loaded via {func}`~hnn_core.law_2021_model()` and
  {func}`~hnn_core.jones_2009_model()`, by [Nick Tolley][] in {gh}`348`

- Add ability to interactivity explore connections in {func}`~hnn_core.viz.plot_cell_connectivity`
  by [Mainak Jas][] in {gh}`376`

- Add {func}`~hnn_core.calcium_model` with a distance dependent calcium channel conductivity, by
  [Nick Tolley][] and [Sarah Pugliese][] in {gh}`348`

- Each drive spike train sampled through an independent process corresponds to a single artificial
  drive cell, the number of which users can set when adding drives with ``n_drive_cells`` and
  ``cell_specific``, by [Ryan Thorpe][] in {gh}`383`

- Add {func}`~hnn_core.pick_connection` to query the indices of specific connections in
  {attr}`~hnn_core.Network.connectivity`, by [Nick Tolley][] in {gh}`367`

- Drives in {attr}`~hnn_core.Network.external_drives` no longer contain a `'conn'` key and the
  {attr}`~hnn_core.Network.connectivity` list contains more items when adding drives from a param
  file or when in legacy mode, by [Ryan Thorpe][], [Mainak Jas][], and [Nick Tolley][] in {gh}`369`

- Add {func}`~hnn_core.optimization.optimize_evoked` to optimize the timing and weights of driving
  inputs for simulating evoked responses, by [Blake Caldwell][] and [Mainak Jas][] in {gh}`77`

- Add method for setting in-plane cell distances and layer separation in the network
  {func}`~hnn_core.Network.set_cell_positions`, by [Christopher Bailey][] in {gh}`370`

- External drives API now accepts probability argument for targeting subsets of cells, by [Nick
  Tolley][] in {gh}`416`

### Bug

- Remove rounding error caused by repositioning of NEURON cell sections, by [Mainak Jas][] and [Ryan
  Thorpe][] in {gh}`314`

- Fix issue where common drives use the same parameters for all cell types, by [Nick Tolley][] in
  {gh}`350`

- Fix bug where depth of L5 and L2 cells were swapped, by [Christopher Bailey][] in {gh}`352`

- Fix bug where {func}`~hnn_core.dipole.average_dipoles` failed when there were less than two
  dipoles in the input dipole list, by [Kenneth Loi][] in {gh}`368`

- Fix bug where {func}`~hnn_core.read_spikes` wasn't returning a {class}`~hnn_core.CellResponse`
  instance with updated spike types, by [Ryan Thorpe][] in {gh}`382`

- {attr}`Dipole.times` and {attr}`Cell_response.times` now reflect the actual integration points
  instead of the intended times, by [Mainak Jas][] in {gh}`397`

- Fix overlapping non-cell-specific drive gid assignment over different ranks in
  {class}`~hnn_core.MPIBackend`, by [Ryan Thorpe][] and [Mainak Jas][] in {gh}`399`

- Allow {func}`~hnn_core.read_dipoles` to read dipole from a file with only two columns (``times``
  and ``data``), by [Mainak Jas][] in {gh}`421`

### API

- New API for defining cell-cell connections. Custom connections can be added with
  {func}`~hnn_core.Network.add_connection`, by [Nick Tolley][] in {gh}`276`

- Remove {class}`~hnn_core.L2Pyr`, {class}`~hnn_core.L5Pyr`, {class}`~hnn_core.L2Basket`, and
  {class}`~hnn_core.L5Basket` classes in favor of instantiation through functions and a more
  consistent {class}`~hnn_core.Cell` class by [Mainak Jas][] in {gh}`322`

- Remove parameter ``distribution`` in {func}`~hnn_core.Network.add_bursty_drive`.  The distribution
  is now Gaussian by default, by [Mainak Jas][] in {gh}`330`

- New API for accessing and modifying {class}`~hnn_core.Cell` attributes (e.g., synapse and
  biophysics parameters) as cells are now instantiated from template cells specified in a
  {class}`~hnn_core.Network` instance's {attr}`~/hnn_core.Network.cell_types` attribute by [Ryan
  Thorpe][] in {gh}`321`

- New API for network creation. The default network is now created with ``net =
  jones_2009_model(params)``, by [Nick Tolley][] in {gh}`318`

- Replace parameter ``T`` with ``tstop`` in {func}`~hnn_core.Network.add_tonic_bias` and
  {func}`~hnn_core.Cell.create_tonic_bias` to be more consistent with other functions and improve
  readability, by [Kenneth Loi][] in {gh}`354`

- Deprecated ``postproc`` argument in {func}`~hnn_core.dipole.simulate_dipole`, whereby user should
  explicitly smooth and scale resulting dipoles, by [Christopher Bailey][] in {gh}`372`

- Number of drive cells and their connectivity can now be specified through the ``n_drive_cells``
  and ``cell_specific`` arguments in ``Network.add_xxx_drive()`` methods, replacing use of
  ``repeats`` and ``sync_within_trial``, by [Ryan Thorpe][] in {gh}`383`

- Simulation end time and integration time have to be specified now with ``tstop`` and ``dt`` in
  {func}`~hnn_core.simulate_dipole`, by [Mainak Jas][] in {gh}`397`

- {meth}`CellResponse.reset` method is not supported any more, by [Mainak Jas][] in {gh}`397`

- Target cell types and their connections are created for each drive according to the synaptic
  weight and delay dictionaries assigned in ``Network.add_xxx_drive()``, by [Ryan Thorpe][] in
  {gh}`369`

- Cell objects can no longer be accessed from {class}`~hnn_core.Network` as the
  {attr}`~hnn_core.Network.cells` attribute has been removed, by [Ryan Thorpe][] in {gh}`436`

### People who contributed to this release (in alphabetical order):

- [Alex Rockhill][]
- [Blake Caldwell][]
- [Christopher Bailey][]
- [Dylan Daniels][]
- [Kenneth Loi][]
- [Mainak Jas][]
- [Nick Tolley][]
- [Ryan Thorpe][]
- [Sarah Pugliese][]
- [Stephanie R. Jones][]

## 0.1

### Changelog

- Add ability to simulate multiple trials in parallel using joblibs, by [Mainak Jas][] in {gh}`44`

- Rhythmic inputs can now be turned off by setting their conductance weights to 0 instead of setting
  their start times to exceed the simulation stop time, by [Ryan Thorpe][] in {gh}`105`

- Reader for parameter files, by [Blake Caldwell][] in {gh}`80`

- Add plotting of voltage at soma to inspect firing pattern of cells, by [Mainak Jas][] in {gh}`86`

- Add ability to simulate a single trial in parallel across cores using MPI, by [Blake Caldwell][]
  in {gh}`79`

- Modify {func}`~hnn_core.viz.plot_dipole` to accept both lists and individual instances of Dipole
  object, by [Nick Tolley][] in {gh}`145`

- Update ``plot_hist_input`` to {func}`~hnn_core.viz.plot_spikes_hist` which can plot histogram of
  spikes for any cell type, by [Nick Tolley][] in {gh}`157`

- Add function to compute mean spike rates with user specified calculation type, by [Nick Tolley][]
  and [Mainak Jas][] in {gh}`155`

- Add ability to record somatic voltages from all cells, by [Nick Tolley][] in {gh}`190`

- Add ability to instantiate external feed event times of a network prior to building it, by
  [Christopher Bailey][] in {gh}`191`

- Add ability to record somatic currents from all cells, by [Nick Tolley][] in {gh}`199`

- Add option to turn off dipole postprocessing, by [Carmen Kohl][] in {gh}`188`

- Add ability to add tonic inputs to cell types with {func}`~hnn_core.Network.add_tonic_bias`, by
  [Mainak Jas][] in {gh}`209`

- Modify {func}`~hnn_core.viz.plot_spikes_raster` to display individual cells, by [Nick Tolley][] in
  {gh}`231`

- Add {meth}`~hnn_core.Network.copy` method for cloning a ``Network`` instance, by [Christopher
  Bailey][] in {gh}`221`

- Add methods for creating input drives and biases to network:
  {meth}`~hnn_core.Network.add_evoked_drive`, {meth}`~hnn_core.Network.add_poisson_drive`,
  {meth}`~hnn_core.Network.add_bursty_drive` and {meth}`~hnn_core.Network.add_tonic_bias`, by
  [Christopher Bailey][] in {gh}`221`

- Add functions for plotting power spectral density ({func}`~hnn_core.viz.plot_psd`) and Morlet
  time-frequency representations ({func}`~hnn_core.viz.plot_tfr_morlet`), by [Christopher Bailey][]
  in {gh}`264`

- Add y-label units (nAm) to all visualisation functions involving dipole moments, by [Christopher
  Bailey][] in {gh}`264`

- Add Savitzky-Golay filtering method {meth}`~hnn_core.dipole.Dipole.savgol_filter` to ``Dipole``;
  copied from ``mne-python`` {meth}`~mne.Evoked.savgol_filter`, by [Christopher Bailey][] in
  {gh}`264`

### Bug

- Fix missing autapses in network construction, by [Mainak Jas][] in {gh}`50`

- Fix rhythmic input feed, by [Ryan Thorpe][] in {gh}`98`

- Fix bug introduced into rhythmic input feed and add test, by [Christopher Bailey][] in {gh}`102`

- Fix bug in amplitude of delay (for connection between L2 Basket and Gaussian feed) being passed
  incorrectly, by [Mainak Jas][] in {gh}`146`

- Connections now cannot be removed by setting the weights to 0., by [Mainak Jas][] and [Ryan
  Thorpe][] in {gh}`162`

- MPI and Joblib backends now apply jitter across multiple trials identically, by [Ryan Thorpe][] in
  {gh}`171`

- Fix bug in Poisson input where the first spike was being missed after the start time, by [Mainak
  Jas][] in {gh}`204`

- Fix bug in network to add empty spike when empty file is read in, by [Samika Kanekar][] and [Ryan
  Thorpe][] in {gh}`207`

### API

- Make a context manager for Network class, by [Mainak Jas][] and [Blake Caldwell][] in {gh}`86`

- Create Spikes class, add write methods and read functions for Spikes and Dipole classes, by [Ryan
  Thorpe][] in {gh}`96`

- Only specify `n_jobs` when instantiating the JoblibBackend, by [Blake Caldwell][] in {gh}`79`

- Make a context manager for parallel backends (JoblibBackend, MPIBackend), by [Blake Caldwell][] in
  {gh}`79`

- Add {func}`~hnn_core.dipole.average_dipoles` function, by [Blake Caldwell][] in {gh}`156`

- New API for defining external drives and biases to network. By default, a
  {class}`~hnn_core.Network` is created without drives, which are added using class methods. The
  argument ``add_drives_from_params`` controls this behaviour, by [Christopher Bailey][] in
  {gh}`221`

- Examples apply random state seeds that reproduce the output of HNN GUI documentation, by
  [Christopher Bailey][] in {gh}`221`

- Force conversion to nAm (from fAm) for output of {func}`~hnn_core.dipole.simulate_dipole`
  regardless of ``postproc``-argument, which now only controls parameter file-based smoothing and
  scaling, by [Christopher Bailey][] in {gh}`264`

### People who contributed to this release (in alphabetical order):

- [Blake Caldwell][]
- [Christopher Bailey][]
- [Carmen Kohl][]
- [Mainak Jas][]
- [Nick Tolley][]
- [Ryan Thorpe][]
- [Samika Kanekar][]
- [Stephanie R. Jones][]

[Alex Rockhill]: https://github.com/alexrockhill
[Blake Caldwell]: https://github.com/blakecaldwell
[Christopher Bailey]: https://github.com/cjayb
[Carmen Kohl]: https://github.com/kohl-carmen
[Dylan Daniels]: https://github.com/dylansdaniels
[Huzi Cheng]: https://github.com/chenghuzi
[Kenneth Loi]: https://github.com/kenloi
[Mainak Jas]: http://jasmainak.github.io/
[Mattan Pelah]: https://github.com/mjpelah
[Mohamed A. Sherif]: https://github.com/mohdsherif/
[Mostafa Khalil]: https://github.com/mkhalil8
[Nick Tolley]: https://github.com/ntolley
[Orsolya Beatrix Kolozsvari]: http://github.com/orbekolo/
[Rajat Partani]: https://github.com/raj1701
[Ryan Thorpe]: https://github.com/rythorpe
[Samika Kanekar]: https://github.com/samikane
[Sarah Pugliese]: https://bcs.mit.edu/directory/sarah-pugliese
[Stephanie R. Jones]: https://github.com/stephanie-r-jones
[Steven Brandt]: https://github.com/spbrandt
[Kaisu Lankinen]: https://github.com/klankinen
[George Dang]: https://github.com/gtdang
[Camilo Diaz]: https://github.com/kmilo9999
[Abdul Samad Siddiqui]: https://github.com/samadpls
[Katharina Duecker]: https://github.com/katduecker
[Yaroslav Halchenko]:  https://github.com/yarikoptic
[Tianqi Cheng]: https://github.com/tianqi-cheng
[Carolina Fernandez Pujol]: https://github.com/carolinafernandezp
[Austin E. Soplata]: https://github.com/asoplata
[Dikshant Jha]: https://github.com/dikshant182004
[Dan Toms]: https://github.com/pynmash
[Shehroz Kashif]: https://github.com/Shehrozkashif
[Mohamed W. ElSayed]: https://github.com/wagdy88
