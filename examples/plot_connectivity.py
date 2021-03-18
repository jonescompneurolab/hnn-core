"""
=====================
07. Plot Connectivity
=====================

This example demonstrates how to modify the network the network connectivity.
"""

# Author: Nick Tolley <nick nicholas_tolley@brown.edu>

import os.path as op

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_params, Network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# To explore how to modify network connectivity, we will start with simulating
# the evoked response from the
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`, and
# explore how it changes with new connections. We first instantiate build
# the network.
net = Network(params)

###############################################################################
# Instantiating the network comes with a predefined set of connections that
# reflect the canonical neocortical microcircuit. ``net.connectivity_list``
# is a list of dictionaries which detail every cell-cell, and drive-cell
# connection.
print(len(net.connectivity_list))
print(net.connectivity_list[0])

###############################################################################
# Next we add the distal drive. Note that adding drives creates new drive-cell
# connections that are appended to ``net.connectivity_list``
weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, seedcore=4)

print(net.connectivity_list[-1])

###############################################################################
# Then we add two proximal drives and simuate the network.
weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}
# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)

# Second proximal evoked drive. NB: only AMPA weights differ from first
weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)

###############################################################################
# Data recorded during simulations are stored under
# :class:`~hnn_core.Cell_Response`. To test multiple network structures, we can
# create a copy of the original network. The copied network is then simulated.
net_erp = net.copy()
dpl_erp = simulate_dipole(net_erp, n_trials=1)
net_erp.cell_response.plot_spikes_raster()

###############################################################################
# We can modify the connectivity list to test the effect of different
# connectivity patterns. For example, we can remove all layer 2 inhibitory
# connections.
new_connectivity = [conn for conn in net.connectivity_list
                    if conn['src_type'] != 'L2Basket']
net.connectivity_list = new_connectivity

net_remove = net.copy()
dpl_remove = simulate_dipole(net_remove, n_trials=1)
net_remove.cell_response.plot_spikes_raster()

###############################################################################
# That's a lot of spiking! We can additionally add new connections using
# ``net.add_connection()``. Let's try connecting a single layer 2 basket cell,
# to every layer 2 pyramidal cell. We can utilize ``net.gid_ranges`` to help
# find the gids of interest.
print(net.gid_ranges)
src_gid = net.gid_ranges['L2_basket'][0]
target_gids = net.gid_ranges['L2_pyramidal']
location, receptor = 'soma', 'gabaa'
delay, weight, lamtha, threshold = 1.0, 1.0, 70, 0.0
for target_gid in target_gids:
    net.add_connection(src_gid, target_gid, location, receptor, delay,
                       weight, lamtha, threshold)

net_add = net.copy()
dpl_add = simulate_dipole(net_add, n_trials=1)
net_add.cell_response.plot_spikes_raster()

###############################################################################
# Adding more inhibitory connections did not completely restore the normal
# spiking. L2 basket and pyramidal cells rhythymically fire in the gamma
# range (30-80 Hz). As a final step, we can see how this change in spiking
# activity impacts the aggregate current dipole.
import matplotlib.pyplot as plt
from viz import plot_dipole
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
dpls = [dpl_erp[0], dpl_remove[0], dpl_add[0]]
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
axes[0].legend(['Normal', 'No L2 Basket', 'Single L2 Basket'])
net_erp.cell_response.plot_spikes_hist(
    ax=axes[1], spike_types=['evprox', 'evdist'])  