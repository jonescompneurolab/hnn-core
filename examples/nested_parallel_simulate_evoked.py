"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using MNE-Neuron.

Run with:

mpiexec --oversubscribe -np 1 python examples/nested_parallel_simulate_evoked.py
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

params_fname = op.join(mne_neuron_root, 'param', 'default.json')

###############################################################################
# Spawn nrniv using MPI

from mpi4py import MPI
from sys import executable, argv

# Start clock
start = MPI.Wtime()

n_trials = 3
n_procs = 8

comm = MPI.COMM_SELF.Spawn('nrniv',
                       args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                             'examples/parallel_simulate_evoked.py', params_fname, str(n_trials)],
                       maxprocs=n_procs, root=0)

comm.Barrier()
comm.Disconnect()
finish = MPI.Wtime() - start
print('\nProcessed in %.2f secs' % finish)

