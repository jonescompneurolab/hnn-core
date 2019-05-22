"""
================================
Calculate simulated dipole error
================================

This example calculates the RMSE between an experimental dipole waveform
and a simulated waveform using MNE-Neuron.
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>


from numpy import loadtxt

import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Params, Network, get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters file

params_fname = op.join(mne_neuron_root, 'param', 'default.json')
params = Params(params_fname)

net = Network(params)


###############################################################################
# Read the dipole data file to compare against

extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

###############################################################################
# Now let's simulate the dipole

from mpi4py import MPI

ntrials = 3
if get_rank() == 0:
    print("Running %d trials" % ntrials)

dpls = []
errs = []
for trial in range(ntrials):
    dpl, err = simulate_dipole(net, trial=trial, inc_evinput=net.params['inc_evinput'],
                           print_progress=False, extdata=extdata)
    dpls.append(dpl)
    errs.append(err)

average_dipoles(dpls).write('avgdpl.txt')

try:
    parent = MPI.Comm.Get_parent()
    parent.Barrier()
    parent.Disconnect()
except MPI.Exception:
    pass

shutdown()

simulate_dipole(net,)
shutdown()