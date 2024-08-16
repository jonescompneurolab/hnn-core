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
   Network
   Cell
   CellResponse
   pick_connection

Network Models (:py:mod:`hnn_core`):
------------------------------------

.. currentmodule:: hnn_core

.. autosummary::
   :toctree: generated/

   jones_2009_model
   law_2021_model
   calcium_model

Optimization (:py:mod:`hnn_core.optimization`):
-----------------------------------------------

.. currentmodule:: hnn_core.optimization

.. autosummary::
   :toctree: generated/

   Optimizer

Dipole (:py:mod:`hnn_core.dipole`):
-----------------------------------

.. currentmodule:: hnn_core.dipole

.. autosummary::
   :toctree: generated/

   Dipole
   average_dipoles

ExtracellularArray (:py:mod:`hnn_core.extracellular`):
------------------------------------------------------

.. currentmodule:: hnn_core.extracellular

.. autosummary::
   :toctree: generated/

   ExtracellularArray

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
   plot_laminar_lfp
   plot_laminar_csd
   NetworkPlotter

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

GUI (:py:mod:`hnn_core.gui`):
-----------------------------

.. currentmodule:: hnn_core.gui

.. autosummary::
   :toctree: generated/

   HNNGUI
