"""
=================================================
08. Optimize simulated evoked response parameters
=================================================

This example demonstrates how to optimize the parameters
of the model simulation to match an experimental dipole waveform.
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import os.path as op

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core.dipole import simulate_dipole
from hnn_core import read_params, Dipole, MPIBackend, Network

hnn_core_root = op.join(op.dirname(hnn_core.__file__))

mpi_cmd = '/autofs/space/meghnn_001/users/mjas/opt/openmpi/bin/mpirun'
n_procs = 10

###############################################################################
# Load experimental data into Dipole object. Data can be retrieved from
# https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/MEG_detection_data/S1_SupraT.txt
#
# This is a different experiment than the one to which the base parameters were
# tuned. So, the initial RMSE will be large, giving the optimization procedure
# a lot to work with.
extdata = np.loadtxt('S1_SupraT.txt')
exp_dpl = Dipole(extdata[:, 0], np.c_[extdata[:, 1],
                 extdata[:, 1], extdata[:, 1]])

###############################################################################
# Read the base parameters from a file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Let's first simulate the dipole with the initial parameters.
net = Network(params, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs, mpi_cmd=mpi_cmd):
    initial_dpl = simulate_dipole(net, n_trials=1)

###############################################################################
# Start the optimization!

from hnn_core.optimization import optimize_evoked

maxiter = 50
with MPIBackend(n_procs=n_procs, mpi_cmd=mpi_cmd):
    params_optim = optimize_evoked(params, exp_dpl, maxiter)

###############################################################################
# Now, let's simulate the dipole with the optimized parameters.
net = Network(params, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs, mpi_cmd=mpi_cmd):
    best_dpl = simulate_dipole(net, n_trials=1)

###############################################################################
# Now plot the results against experimental data:
# 1. Initial dipole
# 2. Optimized dipole fit
#
# Show the input histograms as well

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

exp_dpl.plot(ax=axes[0], layer='agg', show=False)
initial_dpl[0].plot(ax=axes[0], layer='agg', show=False)
best_dpl[0].plot(ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1])
