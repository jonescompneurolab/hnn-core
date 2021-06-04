:orphan:

.. _api_documentation:

=================
API Documentation
=================

Simulation (:py:mod:`hnn_core`):
--------------------------------

.. currentmodule:: hnn_core

.. autosummary::
   :toctree: generated/

   simulate_dipole
   default_network
   Network
   Cell
   CellResponse

Dipole (:py:mod:`hnn_core.dipole`):
-----------------------------------

.. currentmodule:: hnn_core.dipole

.. autosummary::
   :toctree: generated/

   Dipole
   average_dipoles

Visualization (:py:mod:`hnn_core.viz`):
---------------------------------------

.. currentmodule:: hnn_core.viz

.. autosummary::
   :toctree: generated/

   plot_dipole
   plot_spikes_hist
   plot_spikes_raster
   plot_cells
   plot_cell_morphology
   plot_psd
   plot_tfr_morlet
   plot_cell_connectivity
   plot_connectivity_matrix

Parallel backends (:py:mod:`hnn_core.parallel_backends`):
---------------------------------------------------------
.. currentmodule:: hnn_core.parallel_backends

.. autosummary::
   :toctree: generated/

   MPIBackend
   JoblibBackend


Input and Output:
-----------------

.. currentmodule:: hnn_core

.. autosummary::
   :toctree: generated/

   read_params
   read_dipole
   read_spikes
