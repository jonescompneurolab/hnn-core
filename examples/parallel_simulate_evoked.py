"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using MNE-Neuron.

Run with:

mpiexec -np 4 nrniv -python -mpi examples/parallel_simulate_evoked.py
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Params, Network
from mne_neuron import get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters file

params_fname = op.join(mne_neuron_root, 'param', 'default.json')
params = Params(params_fname)

net = Network(params)

###############################################################################
# Now let's simulate the dipole in parallel using all cores

ntrials = 3
if get_rank() == 0:
    print("Running %d trials" % ntrials)

dpls = []
for trial in range(ntrials):
    dpl, err = simulate_dipole(net, trial, net.params['inc_evinput'],
                               verbose=False)
    dpls.append(dpl)

average_dipoles(dpls).write('avgdpl.txt')

shutdown()
