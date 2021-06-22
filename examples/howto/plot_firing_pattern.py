"""
=======================
04. Plot firing pattern
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
from hnn_core import read_params, read_spikes, default_network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Now let's build the network. We have used the same weights as in the
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`.
import matplotlib.pyplot as plt

net = default_network(params)

###############################################################################
# ``net`` does not have any driving inputs and only defines the local network
# connectivity. Let us go ahead and first add a distal evoked drive.
# We need to define the AMPA and NMDA weights for the connections. An
# "evoked drive" defines inputs that are normally distributed with a certain
# mean and standard deviation.

weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': 7e-6,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, seedcore=4)

###############################################################################
# The reason it is called an "evoked drive" is it can be used to simulate
# waveforms resembling evoked responses. Here, we show how to do it with two
# proximal drives which drive current up the dendrite and one distal drive
# which drives current down the dendrite producing the negative deflection.
weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)

###############################################################################
# Now we add the second proximal evoked drive and simulate the network
# dynamics with somatic voltage recordings enabled. Note: only AMPA weights
# differ from first.
weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)

dpls = simulate_dipole(net, record_vsoma=True)

###############################################################################
# Here, we explain more details about the data structures and how they can
# be used to better interpret the data. The cell IDs (gids) uniquely define
# neurons in the network and are stored in the :class:`~hnn_core.Network`
# object as a dictionary
gid_ranges = net.gid_ranges
print(net.gid_ranges)

###############################################################################
# Simulated voltage in the soma is stored in :class:`~hnn_core.CellResponse`
# as a dictionary. The CellResponse object stores data produced by individual
# cells including spikes, somatic voltages and currents.
trial_idx = 0
vsoma = net.cell_response.vsoma[trial_idx]
print(vsoma.keys())

###############################################################################
# We can plot the firing pattern of individual cells by indexing with the gid
gid = 170
plt.figure(figsize=(4, 4), constrained_layout=True)
plt.plot(net.cell_response.times, vsoma[gid])
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
    axes[0].plot(net.cell_response.times, vsoma[gid], color='g')
    gid = gid_ranges['L5_pyramidal'][idx]
    axes[0].plot(net.cell_response.times, vsoma[gid], color='r')
net.cell_response.plot_spikes_raster(ax=axes[1])
net.cell_response.plot_spikes_hist(ax=axes[2],
                                   spike_types=['L5_pyramidal',
                                                'L2_pyramidal'])
