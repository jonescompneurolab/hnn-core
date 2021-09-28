"""
=================================================
05. Optimize simulated evoked response parameters
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
from hnn_core import (MPIBackend, jones_2009_model, read_params,
                      simulate_dipole, read_dipole)


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
exp_dpl = read_dipole('S1_SupraT.txt')

###############################################################################
# Read the base parameters from a file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Let's first simulate the dipole with some initial parameters. The parameter
# definitions also contain the drives. Even though we could add drives
# explicitly through our API
# (see :ref:`sphx_glr_auto_examples_workflows_plot_simulate_evoked.py`),
# for conciseness,
# we add them automatically from the parameter files

scale_factor = 3000.
smooth_window_len = 30.
tstop = exp_dpl.times[-1]
net = jones_2009_model(params=params, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs):
    print("Running simulation with initial parameters")
    initial_dpl = simulate_dipole(net, tstop=tstop, n_trials=1)[0]
    initial_dpl = initial_dpl.scale(scale_factor).smooth(smooth_window_len)

###############################################################################
# Now we start the optimization!

from hnn_core.optimization import optimize_evoked

with MPIBackend(n_procs=n_procs):
    params_optim = optimize_evoked(params, exp_dpl, initial_dpl,
                                   scale_factor=scale_factor,
                                   smooth_window_len=smooth_window_len)

###############################################################################
# Now, let's simulate the dipole with the optimized parameters.
net = jones_2009_model(params=params_optim, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs):
    best_dpl = simulate_dipole(net, tstop=tstop, n_trials=1)[0]
    best_dpl = best_dpl.scale(scale_factor).smooth(smooth_window_len)

###############################################################################
# Finally, we can plot the results against experimental data along with the
# input histograms:
# 1. Initial dipole
# 2. Optimized dipole fit
#
# Upon visualizing the change in optimized versus initial dipole, you should
# consider exploring which parameters were changed to cause the improved dipole
# fit.

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

exp_dpl.plot(ax=axes[0], layer='agg', show=False)
initial_dpl.plot(ax=axes[0], layer='agg', show=False)
best_dpl.plot(ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1])
