"""
==========================
Parallel dipole simulation
==========================

Run with:

mpiexec -np 4 nrniv -python -nobanner -mpi examples/parallel_simulate_evoked.py
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op
from sys import argv
from mpi4py import MPI

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Params, Network, get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Parse command-line arguments

if len(argv) < 8 or not op.exists(argv[7]):
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
else:
    params_fname = argv[7]

if len(argv) < 9:
    ntrials = 1
else:
    try:
        ntrials = int(argv[8])
    except TypeError:
        ntrials = 1

###############################################################################
# Then we read the parameters file

params = Params(params_fname)

net = Network(params)

###############################################################################
# Now let's simulate the dipole

if get_rank() == 0:
    print("Running %d trials" % ntrials)

dpls = []
for trial in range(ntrials):
    dpls.append(simulate_dipole(net, trial, net.params['inc_evinput'], print_progress=False))

average_dipoles(dpls).write('avgdpl.txt')

try:
    parent = MPI.Comm.Get_parent()
    parent.Barrier()
    parent.Disconnect()
except MPI.Exception:
    pass

shutdown()
