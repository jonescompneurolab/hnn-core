"""
=======================================
07. Use MPI backend for parallelization
=======================================

This example demonstrates how to use the MPI backend for
simulating dipoles using HNN-core.

The MPI backend allows running the simulation in parallel across neurons in the
network even with a single trial. For this, you will
need the :ref:`MPI related software <parallel>` installed. Note that if you
want to simulate in parallel across trials, the Joblib backend allows this
without the need to install and configure MPI.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Blake Caldwell <blake_caldwell@brown.edu>

###############################################################################
# Let us import hnn_core
import os.path as op

import hnn_core
from hnn_core import simulate_dipole, read_params, default_network

###############################################################################
# Then we setup the directories and Neuron
hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the default parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
print(params)

###############################################################################
# Update a few of the default parameters related to visualisation
params.update({
    'tstop': 310.0,
})

###############################################################################
# Following :ref:`the alpha example
# <sphx_glr_auto_examples_plot_simulate_alpha.py>`, we add a
# ~10 Hz "bursty" drive starting at 50 ms and continuing to the end of the
# simulation. Each burst consists of a pair (2) of spikes, spaced 10 ms apart.
# The occurrence of each burst is jittered by a random, normally distributed
# amount (20 ms standard deviation). We repeat the burst train 10 times, each
# time with unique randomization.
net = default_network(params)

weights_ampa = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
net.add_bursty_drive(
    'bursty', tstart=50., burst_rate=10, burst_std=20., numspikes=2,
    spike_isi=10, repeats=10, location='distal', weights_ampa=weights_ampa,
    seedcore=4)

###############################################################################
# Finally, to simulate we use the
# :class:`~hnn_core.parallel_backends.MPIBackend` class. This will
# start the simulation across the number of processors (cores) specified by
# ``n_procs`` using MPI. The ``'mpiexec'`` launcher is used from
# ``openmpi``, which must be installed on the system
from hnn_core import MPIBackend

with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(net, n_trials=1, postproc=False)

trial_idx = 0
dpls[trial_idx].plot()
