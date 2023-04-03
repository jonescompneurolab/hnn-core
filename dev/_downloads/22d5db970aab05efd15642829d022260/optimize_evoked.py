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
    net_opt = optimize_evoked(net, tstop=tstop, n_trials=1,
                              target_dpl=exp_dpl, initial_dpl=initial_dpl,
                              scale_factor=scale_factor,
                              smooth_window_len=smooth_window_len)

###############################################################################
# Now, let's simulate the dipole with the optimized drive parameters.
with MPIBackend(n_procs=n_procs):
    best_dpl = simulate_dipole(net_opt, tstop=tstop, n_trials=1)[0]
    best_dpl = best_dpl.scale(scale_factor).smooth(smooth_window_len)

###############################################################################
# Then, we can plot the pre- and post-optimization simulations alongside the
# experimental data. 

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

exp_dpl.plot(ax=axes[0], layer='agg', show=False, color='tab:blue')
initial_dpl.plot(ax=axes[0], layer='agg', show=False, color='tab:orange')
best_dpl.plot(ax=axes[0], layer='agg', show=False, color='tab:green')
axes[0].legend(['experimental', 'initial', 'optimized'])
net_opt.cell_response.plot_spikes_hist(ax=axes[1])

###############################################################################
# Finally, let's explore which parameters were changed to cause the 
# improved dipole fit. 
# As an example, we will look at the new dynamics of the first proximal drive, 
# as well as the synaptic weight of its layer 2/3 pyramidal AMPA receptors.

from hnn_core.network import pick_connection

dynamics_opt = net_opt.external_drives['evdist1']['dynamics']

conn_indices = pick_connection(net=net_opt, src_gids='evprox1', target_gids='L2_pyramidal',
    loc='proximal', receptor='ampa')
conn_idx = conn_indices[0]
weight_opt = net_opt.connectivity[conn_idx]

print("Optimized dynamic properties: ", dynamics_opt)
print("\nOptimized weight: ", weight_opt)

###############################################################################
# Let's compare to the initial dynamic properties and weight.
dynamics_initial = net.external_drives['evdist1']['dynamics']

conn_indices = pick_connection(net=net, src_gids='evprox1', target_gids='L2_pyramidal',
    loc='proximal', receptor='ampa')
conn_idx = conn_indices[0]
weight_initial = net.connectivity[conn_idx]

print("Initial dynamic properties: ", dynamics_initial)
print("\nInitial weight: ", weight_initial)
