"""
=======================
01. Plot firing pattern
=======================

This example demonstrates how to inspect the firing
pattern of cells in the HNN model.
"""

# Authors: Mainak Jas <mjas@harvard.mgh.edu>
#          Nick Tolley <nick nicholas_tolley@brown.edu>

import os.path as op
import tempfile

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_spikes, neymotin_2020_model, simulate_dipole

###############################################################################
# Now let's build the network with the same canonical ERP drives used in the
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`.
import matplotlib.pyplot as plt

net = neymotin_2020_model(load_erp_drives=True)

###############################################################################
# ``load_erp_drives=True`` adds the three default evoked drives (one distal,
# two proximal) used to reproduce the tactile ERP waveform. We then simulate
# the network dynamics with somatic voltage recordings enabled.

dpls = simulate_dipole(net, tstop=170., record_vsec='soma')

###############################################################################
# Here, we explain more details about the data structures and how they can
# be used to better interpret the data. The cell IDs (gids) uniquely define
# neurons in the network and are stored in the :class:`~hnn_core.Network`
# object as a dictionary
gid_ranges = net.gid_ranges
print(net.gid_ranges)

###############################################################################
# Simulated voltage in the soma and other cell sections are stored in
# :class:`~hnn_core.CellResponse` as a dictionary. The CellResponse object
# stores data produced by individual cells including spikes,
# voltages and currents.
trial_idx = 0
vsec = net.cell_response.vsec[trial_idx]
print(vsec.keys())

###############################################################################
# We can plot the firing pattern of individual cells by indexing with the gid
gid = 170
plt.figure(figsize=(4, 4), constrained_layout=True)
plt.plot(net.cell_response.times, vsec[gid]['soma'])
plt.title('%s (gid=%d)' % (net.gid_to_type(gid), gid))
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()

###############################################################################
# Also, we can plot the spikes in the network and write them to text files.
# Note that we can use formatting syntax to specify the filename pattern
# with which each trial will be written ('spk_1.txt', 'spk_2.txt, ...). To
# read spikes back in, we can use wildcard expressions.
net.cell_response.plot_spikes_raster()
with tempfile.TemporaryDirectory() as tmp_dir_name:
    net.cell_response.write(op.join(tmp_dir_name, 'spk_%d.txt'))
    cell_response = read_spikes(op.join(tmp_dir_name, 'spk_*.txt'))
cell_response.plot_spikes_raster()

###############################################################################
# We can additionally calculate the mean spike rates for each cell class by
# specifying a time window with ``tstart`` and ``tstop``.
all_rates = cell_response.mean_rates(tstart=0, tstop=170,
                                     gid_ranges=net.gid_ranges,
                                     mean_type='all')
trial_rates = cell_response.mean_rates(tstart=0, tstop=170,
                                       gid_ranges=net.gid_ranges,
                                       mean_type='trial')
print('Mean spike rates across trials:')
print(all_rates)
print('Mean spike rates for individual trials:')
print(trial_rates)

###############################################################################
# Finally, we can plot the soma voltage along with the spiking activity with
# raster plots and histograms for the pyramidal cells.

fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

for idx in range(10):  # only 10 cells per cell-type
    gid = gid_ranges['L2_pyramidal'][idx]
    axes[0].plot(net.cell_response.times, vsec[gid]['soma'], color='g')
    gid = gid_ranges['L5_pyramidal'][idx]
    axes[0].plot(net.cell_response.times, vsec[gid]['soma'], color='r')
net.cell_response.plot_spikes_raster(ax=axes[1])
net.cell_response.plot_spikes_hist(ax=axes[2],
                                   spike_types=['L5_pyramidal',
                                                'L2_pyramidal'])
