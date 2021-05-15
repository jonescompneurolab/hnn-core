"""
=====================
07. Plot Connectivity
=====================

This example demonstrates how to modify the network connectivity.
"""

# Author: Nick Tolley <nicholas_tolley@brown.edu>

# sphinx_gallery_thumbnail_number = 4

from hnn_core.network import Network
import os.path as op

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_params, default_network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# To explore how to modify network connectivity, we will start with simulating
# the evoked response from the
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`, and
# explore how it changes with new connections. We first instantiate the
# network. (Note: Setting ``add_drives_from_params=True`` loads a set of
# predefined drives without the drives API shown previously).
net_erp = default_network(params, add_drives_from_params=True)

###############################################################################
# Instantiating the network comes with a predefined set of connections that
# reflect the canonical neocortical microcircuit. ``net.connectivity``
# is a list of dictionaries which detail every cell-cell, and drive-cell
# connection.
print(len(net_erp.connectivity))
print(net_erp.connectivity[0:2])

###############################################################################
# Data recorded during simulations are stored under
# :class:`~hnn_core.Cell_Response`. To test multiple network structures, we can
# create a copy of the original network. The copied network is then simulated.
dpl_erp = simulate_dipole(net_erp, n_trials=1)
net_erp.cell_response.plot_spikes_raster()

###############################################################################
# We can also define our own connections to test the effect of different
# connectivity patterns. To start, an empty network is instantiated.
# ``add_drives_from_params`` can still be used with an empty Network, but
# all cells will start disconnected. ``net.add_connection`` is then
# used to create a custom network. Let us first create an all-to-all
# connectivity pattern between the L5 pyramidal cells, and L2 basket cells.
# :meth:`hnn_core.Network.add_connection` allows connections to be specified
# with either cell names, or the gids directly. If multiple gids are provided
# for either the sources or the targets, they will be connected in an
# all-to-all pattern.

net_all = Network(params, add_drives_from_params=True)

# Pyramidal cell connections
location, receptor = 'distal', 'ampa'
weight, delay, lamtha = 1.0, 1.0, 70
src = 'L5_pyramidal'
for target in ['L5_pyramidal', 'L5_pyramidal']:
    net_all.add_connection(src, target, location, receptor,
                           delay, weight, lamtha)

# Basket cell connections
location, receptor = 'soma', 'gabaa'
weight, delay, lamtha = 1.0, 1.0, 70
src = 'L2_basket'
for target in ['L2_basket', 'L5_pyramidal']:
    net_all.add_connection(src, target, location, receptor,
                           delay, weight, lamtha)

dpl_all = simulate_dipole(net_all, n_trials=1)
net_all.cell_response.plot_spikes_raster()

###############################################################################
# That's a lot of spiking! We can additionally use the ``probability``.
# argument to create a sparse connectivity pattern instead of all-to-all. Let's
# try creating the same network with a 10% change of cells connecting
# to each other. The resulting connectivity pattern can also be visualized
# with ``net.connectivity[idx].plot()``
probability = 0.1
net_sparse = Network(params, add_drives_from_params=True)

# Pyramidal cell connections
location, receptor = 'distal', 'ampa'
weight, delay, lamtha = 1.0, 1.0, 70
src = 'L5_pyramidal'
for target in ['L5_pyramidal', 'L2_basket']:
    net_sparse.add_connection(src, target, location, receptor,
                              delay, weight, lamtha, probability)

# Basket cell connections
location, receptor = 'soma', 'gabaa'
weight, delay, lamtha = 1.0, 1.0, 70
src = 'L2_basket'
for target in ['L2_basket', 'L5_pyramidal']:
    net_sparse.add_connection(src, target, location, receptor,
                              delay, weight, lamtha, probability)

dpl_sparse = simulate_dipole(net_sparse, n_trials=1)
net_sparse.cell_response.plot_spikes_raster()
net_sparse.connectivity[-2].plot()
net_sparse.connectivity[-1].plot()

###############################################################################
# Adding a single inhibitory connection didn't completely restored the normal
# spiking. However, layer 2 firing is interrupted at 70 and 120 ms.
# As a final step, we can see how this change in spiking activity impacts
# the aggregate current dipole.
import matplotlib.pyplot as plt
from hnn_core.viz import plot_dipole
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
dpls = [dpl_erp[0], dpl_all[0], dpl_sparse[0]]
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
axes[0].legend(['Default', 'Custom All', 'Custom Sparse'])
net_erp.cell_response.plot_spikes_hist(
    ax=axes[1], spike_types=['evprox', 'evdist'])
