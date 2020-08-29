"""
===================
Plot firing pattern
===================

This example demonstrates how to inspect the firing
pattern of cells in the HNN model.
"""

# Authors: Mainak Jas <mjas@harvard.mgh.edu>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import read_params, Network
from hnn_core.network_builder import NetworkBuilder

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Now let's build the network
import matplotlib.pyplot as plt

net = Network(params)
with NetworkBuilder(net) as network_builder:
    network_builder.cells[0].plot_voltage()

    # The cells are stored in the network object as a list
    cells = network_builder.cells
    print(cells[:5])

    # We have different kinds of cells with different cell IDs (gids)
    gids = [0, 35, 135, 170]
    for gid in gids:
        print(cells[gid].name)

    # We can plot the firing pattern of individual cells
    network_builder.cells[0].plot_voltage()
    plt.title('%s (gid=%d)' % (cells[0].name, gid))

###############################################################################
# Let's do this for the rest of the cell types with a new NetworkBuilder object
with NetworkBuilder(net) as network_builder:
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    for gid, ax in zip([35, 170], axes):
        network_builder.cells[gid].plot_voltage(ax)
        ax.set_title('%s (gid=%d)' % (cells[gid].name, gid))
