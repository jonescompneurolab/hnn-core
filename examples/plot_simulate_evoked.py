"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using HNN-core.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op
import tempfile

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network, read_spikes
from hnn_core.viz import plot_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
print(params)

###############################################################################
# This is a lot of parameters! We can also filter the
# parameters using unix-style wildcard characters
print(params['L2Pyr_soma*'])

###############################################################################
# Let us first create our network from the params file and visualize the cells
# inside it.
net = Network(params)
net.plot_cells()

###############################################################################
# Now let's simulate the dipole, running 2 trials with the Joblib backend.
# To run them in parallel we could set n_jobs to equal the number of trials.
from hnn_core import JoblibBackend

with JoblibBackend(n_jobs=1):
    dpls = simulate_dipole(net, n_trials=2, postproc=True)

###############################################################################
# and then plot it
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
net.plot_spikes_hist(ax=axes[1],
                                   spike_types=['evprox', 'evdist'])

###############################################################################
# Also, we can plot the spikes and write them to txt files.
# Note that we can use formatting syntax to specify the filename pattern
# with which each trial will be written. To read spikes back in, we can use
# wildcard expressions.
net.plot_spikes_raster()
with tempfile.TemporaryDirectory() as tmp_dir_name:
    net.write(op.join(tmp_dir_name, 'spk_%d.txt'))
    cell_response = read_spikes(op.join(tmp_dir_name, 'spk_*.txt'))
print(cell_response[0])
###############################################################################
# We can additionally calculate the mean spike rates for each cell class by
# specifying a time window with tstart and tstop.
all_rates = net.mean_rates(tstart=0, tstop=170,
                                     gid_ranges=net.gid_ranges,
                                     mean_type='all')
trial_rates = net.mean_rates(tstart=0, tstop=170,
                                       gid_ranges=net.gid_ranges,
                                       mean_type='trial')
print('Mean spike rates across trials:')
print(all_rates)
print('Mean spike rates for individual trials:')
print(trial_rates)


###############################################################################
# Now, let us try to make the exogenous driving inputs to the cells
# synchronous and see what happens

params.update({'sync_evinput': True})
net_sync = Network(params)

###############################################################################
# Next, let's simulate a single trial using the MPI backend. This will
# start the simulation trial across the number of processors (cores)
# specified by n_procs using MPI. The 'mpiexec' launcher is for
# openmpi, which must be installed on the system
from hnn_core import MPIBackend

with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
    dpls_sync = simulate_dipole(net_sync, n_trials=1)

dpls_sync[0].plot()
net_sync.plot_spikes_hist()
