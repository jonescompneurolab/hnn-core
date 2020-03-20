"""
===============
Simulate dipole
===============

This example demonstrates how to run simulations with MPI using HNN-core.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network, MPI_backend

hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Create the Network instance to use for simulations
net = Network(params)

###############################################################################
# Now let's simulate the dipole using the MPI backend.
# This will run the mpiexec program on the system (for openmpi)
with MPI_backend(n_procs=2, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(net, n_trials=3)

###############################################################################
# Plot the dipole and spiking results
for dpl in dpls:
    dpl.plot(layer='agg')
net.plot_spikes()
