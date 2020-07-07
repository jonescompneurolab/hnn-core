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
from hnn_core import simulate_dipole, read_params, Network, MPIBackend

hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Create the Network instance to use for simulations
net = Network(params)

###############################################################################
# Now let's simulate the dipole using the MPI backend. This will
# start the simulation across the number of processors (cores)
# specified by n_procs using MPI. The 'mpiexec' launcher is for
# openmpi, which must be installed on the system
with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(net, n_trials=3)

###############################################################################
# Plot the dipole and spiking results
for dpl in dpls:
    dpl.plot(layer='agg')
net.spikes.plot()
