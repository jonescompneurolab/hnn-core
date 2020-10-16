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
from hnn_core import read_params, Network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Now let's build the network
import matplotlib.pyplot as plt

net = Network(params)
dpls = simulate_dipole(net, n_trials=1)

# The cell IDs (gids) are stored in the network object as a dictionary
gid_dict = net.gid_dict
print(net.gid_dict)

# Simulated cell responses are stored in the Spikes object as a dictionary.
trial_idx = 0
vsoma = net.spikes.vsoma[trial_idx]
print(vsoma.keys())

# We can plot the firing pattern of individual cells by indexing with the gid
gid = 170
t_vec = net.spikes.t_vec[trial_idx]
plt.figure(figsize=(4, 4))
plt.plot(t_vec, vsoma[gid])
plt.title('%s (gid=%d)' % (net.gid_to_type(gid), gid))
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()

###############################################################################
# # Let's do this for the rest of the cell types
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
for gid, ax in zip([35, 170], axes):
    ax.plot(t_vec, vsoma[gid])
    ax.set_title('%s (gid=%d)' % (net.gid_to_type(gid), gid))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
plt.show()
