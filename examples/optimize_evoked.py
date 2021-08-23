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
from hnn_core import read_params, Dipole, MPIBackend, jones_2009_model

hnn_core_root = op.join(op.dirname(hnn_core.__file__))

# The number of cores may need modifying depending on your current machine.
n_procs = 10

###############################################################################
# First, we will load experimental data into Dipole object.
#
# This is a different experiment than the one to which the base parameters were
# tuned. So, the initial RMSE will be large, giving the optimization procedure
# a lot to work with.
from urllib.request import urlretrieve

data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
            'data/MEG_detection_data/S1_SupraT.txt')
urlretrieve(data_url, 'S1_SupraT.txt')
extdata = np.loadtxt('S1_SupraT.txt')
exp_dpl = Dipole(extdata[:, 0], np.c_[extdata[:, 1],
                 extdata[:, 1], extdata[:, 1]])

###############################################################################
# Read the base parameters from a file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Let's first simulate the dipole with the initial parameters. The parameter
# definitions also contain the drives. Even though we could add drives
# explicitly through our API (see xxxx example), for conciseness,
# we add them automatically from the parameter files

scale_factor = 3000.
smooth_window_len = 30.
tstop = exp_dpl.times[-1]
net = jones_2009_model(params, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs):
    initial_dpl = simulate_dipole(net, tstop=tstop, n_trials=1)[0]
    initial_dpl = initial_dpl.scale(scale_factor).smooth(smooth_window_len)

###############################################################################
# Start the optimization!

from hnn_core.optimization import optimize_evoked

with MPIBackend(n_procs=n_procs):
    params_optim = optimize_evoked(params, exp_dpl, scale_factor=scale_factor,
                                   smooth_window_len=smooth_window_len)

###############################################################################
# Now, let's simulate the dipole with the optimized parameters.
net = jones_2009_model(params, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs):
    best_dpl = simulate_dipole(net, tstop=tstop, n_trials=1)
    best_dpl[0].scale(scale_factor).smooth(smooth_window_len)

###############################################################################
# Finally, we can plot the results against experimental data:
# 1. Initial dipole
# 2. Optimized dipole fit
#
# Show the input histograms as well

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

exp_dpl.plot(ax=axes[0], layer='agg', show=False)
initial_dpl[0].plot(ax=axes[0], layer='agg', show=False)
best_dpl[0].plot(ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1])
