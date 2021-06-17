"""
===============================
08. Simulate beta modulated ERP
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
# Next a beta event is created by inducing simultaneous proximal
# distal drives.
beta_start = 50.0
weights_ampa_p1 = {'L2_basket': 0.00004, 'L2_pyramidal': 0.00002,
                   'L5_basket': 0.00002, 'L5_pyramidal': 0.00002}
syn_delays_p1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                 'L5_basket': 1.0, 'L5_pyramidal': 1.0}

net.add_bursty_drive(
    'beta_prox', tstart=beta_start, burst_rate=1e-10, burst_std=20.,
    numspikes=2, spike_isi=10, repeats=10, location='proximal',
    weights_ampa=weights_ampa_p1, synaptic_delays=syn_delays_p1, seedcore=14)

# Distal Drive
weights_ampa_d1 = {'L2_basket': 0.00032, 'L2_pyramidal': 0.00008,
                   'L5_pyramidal': 0.00004}
syn_delays_d1 = {'L2_basket': 0.5, 'L2_pyramidal': 0.5,
                 'L5_pyramidal': 0.5}


net.add_bursty_drive(
    'beta_dist', tstart=beta_start + 5.0, burst_rate=1e-10, burst_std=10.,
    numspikes=2, spike_isi=10, repeats=10, location='distal',
    weights_ampa=weights_ampa_d1, synaptic_delays=syn_delays_d1, seedcore=14)

###############################################################################
# And finally we simulate
import matplotlib.pyplot as plt
dpls = simulate_dipole(net, postproc=False)
net.cell_response.plot_spikes_hist()

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1])
