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

   L2Pyr
   L5Pyr
   L2Basket
   L5Basket
   simulate_dipole
   Network
   CellResponse

Dipole (:py:mod:`hnn_core.dipole`):
-----------------------------------

.. currentmodule:: hnn_core.dipole

.. autosummary::
   :toctree: generated/

   Dipole
   simulate_dipole
   read_dipole
   average_dipoles

Params (:py:mod:`hnn_core.params`):
-----------------------------------

.. currentmodule:: hnn_core.params

.. autosummary::
   :toctree: generated/

   Params
   read_params

Visualization (:py:mod:`hnn_core.viz`):
---------------------------------------

.. currentmodule:: hnn_core.viz

.. autosummary::
   :toctree: generated/

   plot_dipole
   plot_spikes_hist
   plot_spikes_raster
   plot_cells
   plot_psd
   plot_tfr_morlet

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

   read_dipole
   read_spikes
