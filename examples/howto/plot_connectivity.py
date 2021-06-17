"""
================================
03. Modifying local connectivity
================================

This example demonstrates how to modify the network connectivity.
"""

# Author: Nick Tolley <nicholas_tolley@brown.edu>

# sphinx_gallery_thumbnail_number = 2

import os.path as op

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_params, jones_2009_model, simulate_dipole

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
net_erp = jones_2009_model(params, add_drives_from_params=True)

###############################################################################
# Instantiating the network comes with a predefined set of connections that
# reflect the canonical neocortical microcircuit. ``net.connectivity``
# is a list of dictionaries which detail every cell-cell, and drive-cell
# connection. The weights of these connections can be visualized with
# :func:`~hnn_core.viz.plot_connectivity_weights` as well as
# :func:`~hnn_core.viz.plot_cell_connectivity`. We can search for specific
# connections using ``net.pick_connection`` which returns the indices
# of ``net.connectivity`` that match the provided parameters.
from hnn_core.viz import plot_connectivity_matrix, plot_cell_connectivity

print(len(net_erp.connectivity))

conn_idx = net_erp.pick_connection(
    src_gids='L5_pyramidal', target_gids='L5_pyramidal', receptor='nmda')[0]
print(net_erp.connectivity[conn_idx])
plot_connectivity_matrix(net_erp, conn_idx)

gid_idx = 11
src_gid = net_erp.connectivity[conn_idx]['src_range'][gid_idx]
fig = plot_cell_connectivity(net_erp, conn_idx, src_gid)

###############################################################################
# Data recorded during simulations are stored under
# :class:`~hnn_core.Cell_Response`. Spiking activity can be visualized after
# a simulation is using :meth:`~hnn_core.Cell_Response.plot_spikes_raster`
dpl_erp = simulate_dipole(net_erp, n_trials=1)
net_erp.cell_response.plot_spikes_raster()

###############################################################################
# We can also define our own connections to test the effect of different
# connectivity patterns. To start, ``net.clear_connectivity()`` can be used
# to clear all cell-to-cell connections. By default, previously defined drives
# to the network are retained, but can be removed with ``net.clear_drives()``.
# ``net.add_connection`` is then used to create a custom network. Let us first
# create an all-to-all connectivity pattern between the L5 pyramidal cells,
# and L2 basket cells. :meth:`hnn_core.Network.add_connection` allows
# connections to be specified with either cell names, or the cell IDs (gids)
# directly.


def get_network(probability=1.0):
    net = jones_2009_model(params, add_drives_from_params=True)
    net.clear_connectivity()

    # Pyramidal cell connections
    location, receptor = 'distal', 'ampa'
    weight, delay, lamtha = 1.0, 1.0, 70
    src = 'L5_pyramidal'
    for target in ['L5_pyramidal', 'L2_basket']:
        net.add_connection(src, target, location, receptor,
                           delay, weight, lamtha, probability=probability)

    # Basket cell connections
    location, receptor = 'soma', 'gabaa'
    weight, delay, lamtha = 1.0, 1.0, 70
    src = 'L2_basket'
    for target in ['L5_pyramidal', 'L2_basket']:
        net.add_connection(src, target, location, receptor,
                           delay, weight, lamtha, probability=probability)
    return net


net_all = get_network()
dpl_all = simulate_dipole(net_all, n_trials=1)

###############################################################################
# We can additionally use the ``probability`` argument to create a sparse
# connectivity pattern instead of all-to-all. Let's try creating the same
# network with a 10% chance of cells connecting to each other.
net_sparse = get_network(probability=0.1)
dpl_sparse = simulate_dipole(net_sparse, n_trials=1)

###############################################################################
# With the previous connection pattern there appears to be synchronous rhythmic
# firing of the L5 pyramidal cells with a period of 10 ms. The synchronous
# activity is visible as vertical lines where several cells fire simultaneously
# Using the sparse connectivity pattern produced a lot more spiking in
# the L5 pyramidal cells.
net_all.cell_response.plot_spikes_raster()
net_sparse.cell_response.plot_spikes_raster()

# Get index of most recently added connection, and a src_gid in src_range.
gid_idx = 5
conn_idx = net_sparse.pick_connection(src_gids='L2_basket')[-1]
src_gid = net_sparse.connectivity[conn_idx]['src_range'][gid_idx]
plot_connectivity_matrix(net_sparse, conn_idx)

conn_idx -= 1
src_gid = net_sparse.connectivity[conn_idx]['src_range'][gid_idx]
plot_connectivity_matrix(net_sparse, conn_idx)

###############################################################################
# Note that the sparsity is in addition to the weight decay with distance
# from the source cell.
src_gid = net_sparse.connectivity[conn_idx]['src_range'][5]
plot_cell_connectivity(net_sparse, conn_idx, src_gid=src_gid)

###############################################################################
# In the sparse network, there still appears to be some rhythmicity
# where the cells are firing synchronously with a smaller period of 4-5 ms.
# As a final step, we can see how this change in spiking activity impacts
# the aggregate current dipole.
import matplotlib.pyplot as plt
from hnn_core.viz import plot_dipole
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)

window_len = 30  # ms
scaling_factor = 3000
dpls = [dpl_erp[0].smooth(window_len).scale(scaling_factor),
        dpl_all[0].smooth(window_len).scale(scaling_factor),
        dpl_sparse[0].smooth(window_len).scale(scaling_factor)]

plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
axes[0].legend(['Default', 'Custom All', 'Custom Sparse'])
net_erp.cell_response.plot_spikes_hist(
    ax=axes[1], spike_types=['evprox', 'evdist'])
