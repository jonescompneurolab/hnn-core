"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using HNN-core.

Run with:

mpiexec -np 4 nrniv -python -mpi examples/parallel_simulate_evoked.py
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate, read_params, Config, mpi

hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)


###############################################################################
# Next we build the configuration
config = Config(params)

###############################################################################
# Now let's simulate the dipole in parallel using all cores
dpls, spks = simulate(config, n_trials=2, n_jobs=1)

mpi.shutdown()
