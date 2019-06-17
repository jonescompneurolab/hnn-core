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


from numpy import loadtxt, mean
import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Params, Network
from mne_neuron import get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Try to read the parameters and exp data via MPI

from mpi4py import MPI

try:
    comm = MPI.Comm.Get_parent()

    # receive params and extdata
    (params, extdata) = comm.bcast(comm.Get_rank(), root=0)

    # if run by MPI, suppress output
    verbose = False

###############################################################################
# Otherwise read the params and exp file from disk

except MPI.Exception:

    # Have to read the parameters from a file
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    print("Reading parameters from file:", params_fname)
    params = Params(params_fname)

    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

    verbose = True

###############################################################################
# Build our Network and set up parallel simulation

net = Network(params)

###############################################################################
# Get number of trials

try:
    ntrials = net.params['N_trials']
except KeyError:
    ntrials = 1

###############################################################################
# Now let's simulate the dipole

if get_rank() == 0 and verbose:
    print("Running %d trials" % ntrials)

dpls = []
errs = []
for trial in range(ntrials):
    dpl, err = simulate_dipole(net, trial=trial,
                               inc_evinput=net.params['inc_evinput'],
                               verbose=False, extdata=extdata)
    dpls.append(dpl)
    errs.append(err)

if get_rank() == 0:
    avg_rmse = mean(errs)
    if verbose:
       print("Avg. RMSE:", avg_rmse)

try:
    if get_rank() == 0:
        # send results back to parent
        comm.send((average_dipoles(dpls), avg_rmse), dest=0)

    comm.Barrier()
    comm.Disconnect()

except NameError:
    # don't fail if this script was called without MP`
    pass

shutdown()
