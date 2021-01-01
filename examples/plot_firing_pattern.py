"""
===================
Plot firing pattern
===================

This example demonstrates how to inspect the firing
pattern of cells in the HNN model.
"""

# Authors: Mainak Jas <mjas@harvard.mgh.edu>
#          Nick Tolley <nick nicholas_tolley@brown.edu>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import read_params, Network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Now let's build the network with somatic voltage recordings enabled
import matplotlib.pyplot as plt

net = Network(params)
dpls = simulate_dipole(net, n_trials=1, record_vsoma=True)

###############################################################################
# The cell IDs (gids) are stored in the network object as a dictionary
gid_ranges = net.gid_ranges
print(net.gid_ranges)

###############################################################################
# Simulated voltage in the soma is stored in CellResponse.
# The CellResponse object stores data produced by individual cells including
# spikes, somatic voltages and currents.
trial_idx, gid = 0, 170
vsoma = net.cell_response[gid].vsoma[trial_idx]

###############################################################################
# We can plot the firing pattern of individual cells by indexing with the gid
gid = 170
plt.figure(figsize=(4, 4))
plt.plot(net.times, net.cell_response[gid].vsoma[trial_idx])
plt.title('%s (gid=%d)' % (net.gid_to_type(gid), gid))
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()

###############################################################################
# Let's plot the soma voltage along with the spiking activity with raster
# plots and histograms for the Pyramidal cells.

fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

for idx in range(10):  # only 10 cells per cell-type
    gid = gid_ranges['L2_pyramidal'][idx]
    axes[0].plot(net.times, net.cell_response[gid].vsoma[trial_idx], color='g')
    gid = gid_ranges['L5_pyramidal'][idx]
    axes[0].plot(net.times, net.cell_response[gid].vsoma[trial_idx], color='r')
net.plot_spikes_raster(ax=axes[1])
net.plot_spikes_hist(ax=axes[2], spike_types=['L5_pyramidal', 'L2_pyramidal'])
