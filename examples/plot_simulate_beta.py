"""
===============================
09. Simulate beta modulated ERP
===============================

This example demonstrates how event related potentials (ERP) are modulated
by prestimulus beta events.
"""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

###############################################################################
# Let us import hnn_core
import hnn_core
from hnn_core import simulate_dipole, law_model
from hnn_core.viz import plot_dipole

###############################################################################
# We begin by instantiating the network model described in Law et al. 2021.
net = law_model()

###############################################################################
# Next a beta event is created by inducing simultaneous proximal. Strong drive
# to the L2 basket cells force them to spike, leading to a GABAb mediated
# inhibition of the network, and ultimately suppressing sensory detection.
beta_start = 50.0
# Check weight on L2 basket from paper
weights_ampa_p1 = {'L2_basket': 0.0004, 'L2_pyramidal': 0.00002,
                   'L5_basket': 0.00002, 'L5_pyramidal': 0.00002}
syn_delays_p1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                 'L5_basket': 1.0, 'L5_pyramidal': 1.0}

net.add_bursty_drive(
    'beta_prox', tstart=beta_start, tstart_std=0., tstop=beta_start + 5.,
    burst_rate=1e-10, burst_std=20., numspikes=2, spike_isi=10, repeats=10,
    location='proximal', weights_ampa=weights_ampa_p1,
    synaptic_delays=syn_delays_p1, seedcore=14)

# Distal Drive
weights_ampa_d1 = {'L2_basket': 0.0032, 'L2_pyramidal': 0.00008,
                   'L5_pyramidal': 0.00004}
syn_delays_d1 = {'L2_basket': 0.5, 'L2_pyramidal': 0.5,
                 'L5_pyramidal': 0.5}


net.add_bursty_drive(
    'beta_dist', tstart=beta_start, tstart_std=15., tstop=beta_start + 5.,
    burst_rate=1e-10, burst_std=10., numspikes=2, spike_isi=10, repeats=10,
    location='distal', weights_ampa=weights_ampa_d1,
    synaptic_delays=syn_delays_d1, seedcore=14)

###############################################################################
# To demonstrate sensory depression, we will add an ERP similar to
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`.
# but modified to reflect that parameters used in Law et al. 2021.

stimulus_start = 150.0

# Distal evoked drive
weights_ampa_d1 = {'L2_basket': 0.0005, 'L2_pyramidal': 0.004,
                   'L5_pyramidal': 0.0005}
weights_nmda_d1 = {'L2_basket': 0.0005, 'L2_pyramidal': 0.004,
                   'L5_pyramidal': 0.0005}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=70.0 + stimulus_start, sigma=0.0, numspikes=1,
    weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
    location='distal', synaptic_delays=synaptic_delays_d1, seedcore=4)

# Two proximal drives
weights_ampa_p1 = {'L2_basket': 0.002, 'L2_pyramidal': 0.0011,
                   'L5_basket': 0.001, 'L5_pyramidal': 0.001}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=25.0 + stimulus_start, sigma=0.0, numspikes=1,
    weights_ampa=weights_ampa_p1, weights_nmda=None,
    location='proximal', synaptic_delays=synaptic_delays_prox, seedcore=4)

# Second proximal evoked drive. NB: only AMPA weights differ from first
weights_ampa_p2 = {'L2_basket': 0.005, 'L2_pyramidal': 0.005,
                   'L5_basket': 0.01, 'L5_pyramidal': 0.01}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=135.0 + stimulus_start, sigma=0.0, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, seedcore=4)


###############################################################################
# And finally we simulate
import matplotlib.pyplot as plt
dpls = simulate_dipole(net, postproc=False)
dpls_orig = dpls[0].copy()
dpls_smooth = dpls[0].smooth(45)
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10),
                         constrained_layout=True)
plot_dipole(dpls_orig, ax=axes[0], layer='agg', tmin=1.0, show=False)
plot_dipole(dpls_smooth, ax=axes[0], layer='agg', tmin=1.0, show=False)
net.cell_response.plot_spikes_hist(ax=axes[1], show=False)
net.cell_response.plot_spikes_raster(ax=axes[2], show=False)
