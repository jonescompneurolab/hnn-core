"""
=====================
06. Plot Connectivity
=====================

This example demonstrates how to modify the network connectivity.
"""

# Author: Nick Tolley <nicholas_tolley@brown.edu>

# sphinx_gallery_thumbnail_number = 2

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
# connection. The weights of these connections can be visualized with
# :func:`~hnn_core.viz.plot_connectivity_weights` as well as
# :func:`~hnn_core.viz.plot_cell_connectivity`
from hnn_core.viz import plot_connectivity_matrix, plot_cell_connectivity
print(len(net_erp.connectivity))

conn_idx = 20
print(net_erp.connectivity[conn_idx])
plot_connectivity_matrix(net_erp, conn_idx)

gid_idx = 11
src_gid = net_erp.connectivity[conn_idx]['src_range'][gid_idx]
fig, ax = plot_cell_connectivity(net_erp, conn_idx, src_gid)

###############################################################################
# Data recorded during simulations are stored under
# :class:`~hnn_core.Cell_Response`. Spiking activity can be visualized after
# a simulation is using :meth:`~hnn_core.Cell_Response.plot_spikes_raster`
dpl_erp = simulate_dipole(net_erp, n_trials=1)
net_erp.cell_response.plot_spikes_raster()

###############################################################################
# We can also define our own connections to test the effect of different
# connectivity patterns. To start, ``net.clear_connectivity()`` can be used
# to clear all cell to cell connections. By default, previously defined drives
# to the network are retained, but can be removed with ``net.clear_drives()``.
# ``net.add_connection`` is then used to create a custom network. Let us first
# create an all-to-all connectivity pattern between the L5 pyramidal cells,
# and L2 basket cells. :meth:`hnn_core.Network.add_connection` allows
# connections to be specified with either cell names, or the gids directly.
# If multiple gids are provided for either the sources or the targets,
# they will be connected in an all-to-all pattern.

net_all = default_network(params, add_drives_from_params=True)
net_all.clear_connectivity()

# Pyramidal cell connections
location, receptor = 'distal', 'ampa'
weight, delay, lamtha = 1.0, 1.0, 70
src = 'L5_pyramidal'
for target in ['L5_pyramidal', 'L2_basket']:
    net_all.add_connection(src, target, location, receptor,
                           delay, weight, lamtha)

# Basket cell connections
location, receptor = 'soma', 'gabaa'
weight, delay, lamtha = 1.0, 1.0, 70
src = 'L2_basket'
for target in ['L5_pyramidal', 'L2_basket']:
    net_all.add_connection(src, target, location, receptor,
                           delay, weight, lamtha)

dpl_all = simulate_dipole(net_all, n_trials=1)
net_all.cell_response.plot_spikes_raster()

###############################################################################
# With the previous connection pattern there appears to be synchronous rhythmic
# firing of the L5 pyramidal cells with a period of 10 ms. The synchronous
# activity is visible as vertical lines where several cells fire simultaneously
# We can additionally use the ``probability``. argument to create a sparse
# connectivity pattern instead of all-to-all. Let's try creating the same
# network with a 10% chance of cells connecting to each other.
probability = 0.1
net_sparse = default_network(params, add_drives_from_params=True)
net_sparse.clear_connectivity()

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
for target in ['L5_pyramidal', 'L2_basket']:
    net_sparse.add_connection(src, target, location, receptor,
                              delay, weight, lamtha, probability)

dpl_sparse = simulate_dipole(net_sparse, n_trials=1)
net_sparse.cell_response.plot_spikes_raster()

# Get index of most recently added connection, and a src_gid in src_range.
conn_idx, gid_idx = len(net_sparse.connectivity) - 1, 5
src_gid = net_sparse.connectivity[conn_idx]['src_range'][gid_idx]
plot_connectivity_matrix(net_sparse, conn_idx)
plot_cell_connectivity(net_sparse, conn_idx, src_gid)

conn_idx, gid_idx = len(net_sparse.connectivity) - 2, 5
src_gid = net_sparse.connectivity[conn_idx]['src_range'][gid_idx]
plot_connectivity_matrix(net_sparse, conn_idx)
plot_cell_connectivity(net_sparse, conn_idx, src_gid)

###############################################################################
# Using the sparse connectivity pattern produced a lot more spiking in
# the L5 pyramidal cells. Nevertheless there appears to be some rhythmicity
# where the cells are firing synchronously with a smaller period of 4-5 ms.
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
