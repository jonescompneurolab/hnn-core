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
