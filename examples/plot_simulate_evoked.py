"""
=====================================
01. Simulate dipole for evoked inputs
=====================================

This example demonstrates how to simulate a dipole for evoked-like
waveforms using HNN-core.

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>

# sphinx_gallery_thumbnail_number = 3

import os.path as op
import tempfile

import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, default_network
from hnn_core.viz import plot_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
print(params)

###############################################################################
# This is a lot of parameters! We can also filter the
# parameters using unix-style wildcard characters. Most of the parameters
# relate to within-network connections and the cell geometry.
print(params['L2Pyr_soma*'])

###############################################################################
# Let us first create our network from the params file and visualize the cells
# inside it.
net = default_network(params)
net.plot_cells()
net.cell_types['L5_pyramidal'].plot_morphology()

###############################################################################
# The network of cells is now defined, to which we add external drives as
# required. Weights are prescribed separately for AMPA and NMDA receptors
# (receptors that are not used can be omitted or set to zero). The possible
# drive types include the following (click on the links for documentation):
#
# - :meth:`hnn_core.Network.add_evoked_drive`
# - :meth:`hnn_core.Network.add_poisson_drive`
# - :meth:`hnn_core.Network.add_bursty_drive`

###############################################################################
# First, we add a distal evoked drive
weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, seedcore=4)

###############################################################################
# Then, we add two proximal drives
weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}
# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)

# Second proximal evoked drive. NB: only AMPA weights differ from first
weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)

###############################################################################
# Now let's simulate the dipole, running 2 trials with the
# :class:`~hnn_core.parallel_backends.Joblib` backend.
# To run them in parallel we could set ``n_jobs`` to equal the number of
# trials. The ``Joblib`` backend allows running the simulations in parallel
# across trials.
from hnn_core import JoblibBackend

with JoblibBackend(n_jobs=1):
    dpls = simulate_dipole(net, n_trials=2, postproc=True)

###############################################################################
# and then plot the amplitudes of the simulated aggregate dipole moments over
# time
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1],
                                   spike_types=['evprox', 'evdist'])

###############################################################################
# Now, let us try to make the exogenous driving inputs to the cells
# synchronous and see what happens. This is achieved by setting the parameter
# ``sync_within_trial`` to ``True``. Using the ``copy``-method, we can create
# a clone of the network defined above, and then modify the drive dynamics for
# each drive. Making a copy removes any existing outputs from the network
# such as spiking information and voltages at the soma.

net_sync = net.copy()
net_sync.external_drives['evdist1']['dynamics']['sync_within_trial'] = True
net_sync.external_drives['evprox1']['dynamics']['sync_within_trial'] = True
net_sync.external_drives['evprox2']['dynamics']['sync_within_trial'] = True

###############################################################################
# You may interrogate current values defining the spike event time dynamics by
print(net_sync.external_drives['evdist1']['dynamics'])

###############################################################################
# Finally, let's simulate this network.
dpls_sync = simulate_dipole(net_sync, n_trials=1)

trial_idx = 0
dpls_sync[trial_idx].plot()
net_sync.cell_response.plot_spikes_hist()
