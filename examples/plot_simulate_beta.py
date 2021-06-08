"""
===============================
08. Simulate beta modulated ERP
===============================

This example demonstrates how event related potentials (ERP) are modulated
by prestimulus beta events.
"""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, beta_erp_network

###############################################################################
# Then we setup the directories and read the default parameters file
hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
print(params)

###############################################################################
# We begin by instantiating the network model described in Law et al. 2021.
net = beta_erp_network(params)

###############################################################################
# Next a beta event is created by inducing simultaneous proximal
# distal drives.

weights_ampa_p1 = {'L2_basket': 0.00004, 'L2_pyramidal': 0.00002,
                   'L5_basket': 0.00002, 'L5_pyramidal': 0.00002}
syn_delays_p1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_basket': 1.0, 'L5_pyramidal': 1.0}
# What spike ISI??
net.add_bursty_drive(
    'beta_prox', tstart=40., burst_rate=10, burst_std=20., numspikes=2,
    spike_isi=1, repeats=10, location='proximal', weights_ampa=weights_ampa_p1,
    synaptic_delays=syn_delays_p1, seedcore=14)

# Distal Drive
weights_ampa_d1 = {'L2_basket': 0.00032, 'L2_pyramidal': 0.00008,
                   'L5_pyramidal': 0.00004}
syn_delays_d1 = {'L2_basket': 0.5, 'L2_pyramidal': 0.5,
                      'L5_pyramidal': 0.5}

net.add_bursty_drive(
    'beta_dist', tstart=40., burst_rate=10, burst_std=10., numspikes=2,
    spike_isi=1, repeats=10, location='distal', weights_ampa=weights_ampa_d1,
    synaptic_delays=syn_delays_d1, seedcore=14)
