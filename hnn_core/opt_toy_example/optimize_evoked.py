"""
=================================================
05. Optimize simulated evoked response parameters
=================================================

This example demonstrates how to optimize the parameters
of the model simulation to match an experimental dipole waveform.
"""

# Authors:

import os.path as op

import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import (MPIBackend, jones_2009_model, simulate_dipole,
                      read_dipole)

hnn_core_root = op.join(op.dirname(hnn_core.__file__))

# The number of cores may need modifying depending on your current machine.
n_procs = 2

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
# Now we start the optimization!
#
# First, we define a function that will tell the optimization routine how to
# modify the network drive parameters. The function will take in the Network
# object with no attached drives, and a dictionary of the paramters we wish to
# optimize.


def set_params(net, params_dict):

    # Proximal 1
    weights_ampa_p1 = {'L2_basket': params_dict['evprox1_ampa_L2_basket'],
                       'L2_pyramidal': params_dict['evprox1_ampa_L2_pyramidal'],
                       'L5_basket': params_dict['evprox1_ampa_L5_basket'],
                       'L5_pyramidal': params_dict['evprox1_ampa_L5_pyramidal']}
    synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

    net.add_evoked_drive('evprox1',
                         mu=params_dict['evprox1_mu'],
                         sigma=params_dict['evprox1_sigma'],
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa_p1,
                         synaptic_delays=synaptic_delays_p)

    # Distal
    weights_ampa_d1 = {'L2_basket': params_dict['evdist1_ampa_L2_basket'],
                       'L2_pyramidal': params_dict['evdist1_ampa_L2_pyramidal'],
                       'L5_pyramidal': params_dict['evdist1_ampa_L5_pyramidal']}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}

    net.add_evoked_drive('evdist1',
                         mu=params_dict['evdist1_mu'],
                         sigma=params_dict['evdist1_sigma'],
                         numspikes=1,
                         location='distal',
                         weights_ampa=weights_ampa_d1,
                         weights_nmda=weights_nmda_d1,
                         synaptic_delays=synaptic_delays_d1)

    # Proximal 2
    weights_ampa_p2 = {'L2_basket': params_dict['evprox2_ampa_L2_basket'],
                       'L2_pyramidal': params_dict['evprox2_ampa_L2_pyramidal'],
                       'L5_basket': params_dict['evprox2_ampa_L5_basket'],
                       'L5_pyramidal': params_dict['evprox2_ampa_L5_pyramidal']}

    net.add_evoked_drive('evprox2',
                         mu=params_dict['evprox2_mu'],
                         sigma=params_dict['evprox2_sigma'],
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa_p2,
                         synaptic_delays=synaptic_delays_p)

    return net

###############################################################################
# Then, we define the constrainst.
#
# The constraints must be a dictionary of tuples where the first value in each
# tuple is the lower bound and the second value is the upper bound for the
# corresponding parameter.


constraints = dict()
constraints.update({'evprox1_ampa_L2_basket': (0.01, 1.),
                    'evprox1_ampa_L2_pyramidal': (0.01, 1.),
                    'evprox1_ampa_L5_basket': (0.01, 1.),
                    'evprox1_ampa_L5_pyramidal': (0.01, 1.),
                    'evprox1_mu': (5., 50.),
                    'evprox1_sigma': (2., 25.),
                    'evdist1_ampa_L2_basket': (0.01, 1.),
                    'evdist1_ampa_L2_pyramidal': (0.01, 1.),
                    'evdist1_ampa_L5_pyramidal': (0.01, 1.),
                    'evdist1_mu': (50., 80.),
                    'evdist1_sigma': (2., 25.),
                    'evprox2_ampa_L2_basket': (0.01, 1.),
                    'evprox2_ampa_L2_pyramidal': (0.01, 1.),
                    'evprox2_ampa_L5_basket': (0.01, 1.),
                    'evprox2_ampa_L5_pyramidal': (0.01, 1.),
                    'evprox2_mu': (125., 150.),
                    'evprox2_sigma': (10., 60.)})


###############################################################################
# Now we define and fit the optimizer.

from general import Optimizer

tstop = exp_dpl.times[-1]
scale_factor = 3000
smooth_window_len = 2

net = jones_2009_model()
optim = Optimizer(net, constraints, set_params, solver='cobyla',
                  obj_fun='evoked', scale_factor=scale_factor,
                  smooth_window_len=smooth_window_len, tstop=tstop)
optim.fit(exp_dpl.data['agg'])

###############################################################################
# Finally, we can plot the experimental data alongside the post-optimization
# simulation dipole as well as the convergence plot.

with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
    opt_dpl = simulate_dipole(optim.net_, tstop=tstop, n_trials=1)[0]
opt_dpl.scale(scale_factor)
opt_dpl.smooth(smooth_window_len)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

exp_dpl.plot(ax=axes[0], layer='agg', show=False, color='tab:blue')
opt_dpl.plot(ax=axes[0], layer='agg', show=False, color='tab:green')
axes[0].legend(['experimental', 'optimized'])
optim.net_.cell_response.plot_spikes_hist(ax=axes[1])

fig1 = optim.plot_convergence()