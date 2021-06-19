"""
===================================
02. Record extracellular potentials
===================================

The main output of HNN simulations is the 'dipole' waveform, i.e., the net
intracellular current flowing in pyramidal cell apical dendrites. At the large
distances between cells and M/EEG sensors, this 'primary' current is the main
contributor to the measured fields. Close to the cells, the local field
potential (LFP) is the result of intracellular current leaking into the
extracellular medium through active and passive membrane channels. Under some
simplifying assumptions, we may approximate the LFP at virtual electrodes
placed in and around the HNN network model.
"""

# Authors: Christopher Bailey <cjb@cfin.au.dk>
#          Mainak Jas <mainakjas@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>

# sphinx_gallery_thumbnail_number = 3

import os.path as op
import matplotlib.pyplot as plt

###############################################################################
# We will use the default network with three evoked drives; see
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>` for
# details. We'll go ahead and use the drive features defined in the parameter
# file.

import hnn_core
from hnn_core import read_params, jones_2009_model, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
net = jones_2009_model(params, add_drives_from_params=True)

###############################################################################
# Extracellular recordings require specifying the electrode postions. It can be
# useful to visualize the cells of the network to decide on the placement of
# each electrode.
net.plot_cells()

###############################################################################
# The default network consists of 2 layers (L2 and L5), within which the cell
# somas are arranged in a regular grid, and apical dendrites are aligned along
# the z-axis. We can simulate a linear multielectrode array with 100 um
# intercontact spacing [1]_ by specifying a list of (x, y, z) coordinate
# triplets. The L5 pyramidal cell somas are at z=0 um, with apical dendrites
# extending up to approximately z=2000 um. L2 pyramidal cell somas reside at
# z=1300 um, and have apical dendrites extending to z=2300 um. We'll place the
# recording array in the center of the network. By default, a value of
# 0.3 S/m is used for the constant extracellular conductivity and the
# 'point source approximation' for calculations; see
# :meth:`hnn_core.Network.add_electrode_array` for details.

depths = list(range(-325, 2150, 25))
electrode_pos = [(135, 135, dep) for dep in depths]
net.add_electrode_array('shank1', electrode_pos)

###############################################################################
# The electrode arrays are stored under ``Network.rec_arrays`` as a dictionary
# of :class:`hnn_core.extracellular.ElectrodeArray` objects that are now
# attached to the network and will be recorded during the simulation. Note that
# calculating the extracellular potentials requires additional computational
# resources and will thus slightly slow down the simulation.
# :ref:`Using MPI <sphx_glr_auto_examples_plot_simulate_mpi_backend.py>` will
# speed up computation considerably. Note that we will perform smoothing of the
# dipole time series during plotting (``postproc=False``)
print(net.rec_arrays)
net.plot_cells()

dpl = simulate_dipole(net, tstop=170)

###############################################################################
# For plotting both aggregate dipole moment and LFP traces, we'll use a 10 ms
# smoothing window, after which both data can be decimated by a factor of 20
# from 40 to 2 kHz sampling rates (note that decimation is applied in two
# steps). Decimation speeds up plotting significantly.
trial_idx = 0
window_len = 10  # ms
decimate = [5, 4]  # from 40k to 8k to 2k
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 8),
                        gridspec_kw={'height_ratios': [1, 3, 2]})

# Then plot the aggregate dipole time series on its own axis
dpl[trial_idx].copy().smooth(
    window_len=window_len).plot(ax=axs[0], decim=decimate,
                                show=False)

voltage_offset = 50  # the spacing between individual traces
voltage_scalebar = 200  # can be different from offset
# we can assign each electrode a unique color using a linear colormap
colors = plt.get_cmap('cividis', len(electrode_pos))
# --> XXX just temporary!
labels = []
for lidx, dep in enumerate(depths):
    labels += [dep] if not lidx % 4 else ['']
# <-- XXX
# use the same smoothing window on the LFP traces to allow comparison to dipole
net.rec_arrays['shank1'][trial_idx].smooth(window_len=window_len).plot(
    ax=axs[1], contact_labels=labels, color=colors, decim=decimate, show=False,
    voltage_offset=voltage_offset, voltage_scalebar=voltage_scalebar)

axs[1].grid(True, which='major', axis='x')
axs[1].set_xlabel('')
# Finally, add the spike raster to the bottom subplot
net.cell_response.plot_spikes_raster(ax=axs[2], show=False)
plt.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] Kajikawa, Y. & Schroeder, C. E. How local is the local field
#        potential? Neuron 72, 847–858 (2011).
