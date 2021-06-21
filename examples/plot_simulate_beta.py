"""
===============================
09. Simulate beta modulated ERP
===============================

This example demonstrates how event related potentials (ERP) are modulated
by prestimulus beta events.
"""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

###############################################################################
# Importing the needed functions
from hnn_core import simulate_dipole, law_2021_model
from hnn_core.viz import plot_dipole

###############################################################################
# We begin by instantiating the network model described in Law et al. 2021.
# The model can be instantiated in the same pattern as ``default_network``
# using ``law_model(params)`` as in previous exmaples. Leaving the arguments
# empty loads the default parameter set ``default.json``, and modifies it
# according to Law et al. 2021 [1]_ :
# - the rise and fall time constants of GABAB-conductances on L2 and L5
#   pyramidal cells are _greatly_ increased
# - several synaptic weights are adjusted
# - the total simulation is extended to 400 ms
net = law_2021_model()

###############################################################################
# To demonstrate sensory depression, we will add an ERP similar to
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`,
# but modified to reflect the parameters used in Law et al. 2021.
# Specifically, we are considering the case where a tactile stimulus is
# delivered at 150 ms. 25 ms later, the first input to sensory cortex arrives
# as a proximal drive, followed by one distal and a final late proximal drive.
def add_erp_drives(net, stimulus_start):
    # Distal evoked drive
    weights_ampa_d1 = {'L2_basket': 0.0005, 'L2_pyramidal': 0.004,
                       'L5_pyramidal': 0.0005}
    weights_nmda_d1 = {'L2_basket': 0.0005, 'L2_pyramidal': 0.004,
                       'L5_pyramidal': 0.0005}
    syn_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        'evdist1', mu=70.0 + stimulus_start, sigma=0.0, numspikes=1,
        weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
        location='distal', synaptic_delays=syn_delays_d1, seedcore=4)

    # Two proximal drives
    weights_ampa_p1 = {'L2_basket': 0.002, 'L2_pyramidal': 0.0011,
                       'L5_basket': 0.001, 'L5_pyramidal': 0.001}
    syn_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                       'L5_basket': 1., 'L5_pyramidal': 1.}

    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
        'evprox1', mu=25.0 + stimulus_start, sigma=0.0, numspikes=1,
        weights_ampa=weights_ampa_p1, weights_nmda=None,
        location='proximal', synaptic_delays=syn_delays_prox, seedcore=4)

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {'L2_basket': 0.005, 'L2_pyramidal': 0.005,
                       'L5_basket': 0.01, 'L5_pyramidal': 0.01}
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        'evprox2', mu=135.0 + stimulus_start, sigma=0.0, numspikes=1,
        weights_ampa=weights_ampa_p2, location='proximal',
        synaptic_delays=syn_delays_prox, seedcore=4)

    return net

###############################################################################
# Next a beta event is created by inducing simultaneous proximal and distal
# drives. The input is just strong enough to evoke spiking in the
# L2 basket cells. This spiking causes GABAb mediated inhibition
# of the network, and ultimately suppressed sensory detection.
def add_beta_drives(net, beta_start):
    # Distal Drive
    weights_ampa_d1 = {'L2_basket': 0.00032, 'L2_pyramidal': 0.00008,
                       'L5_pyramidal': 0.00004}
    syn_delays_d1 = {'L2_basket': 0.5, 'L2_pyramidal': 0.5,
                     'L5_pyramidal': 0.5}
    net.add_bursty_drive(
        'beta_dist', tstart=beta_start, tstart_std=0., tstop=beta_start + 50.,
        burst_rate=1., burst_std=10., numspikes=2, spike_isi=10, repeats=10,
        location='distal', weights_ampa=weights_ampa_d1,
        synaptic_delays=syn_delays_d1, seedcore=2)

    # Proximal Drive
    weights_ampa_p1 = {'L2_basket': 0.00004, 'L2_pyramidal': 0.00002,
                       'L5_basket': 0.00002, 'L5_pyramidal': 0.00002}
    syn_delays_p1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_basket': 1.0, 'L5_pyramidal': 1.0}

    net.add_bursty_drive(
        'beta_prox', tstart=beta_start, tstart_std=0., tstop=beta_start + 50.,
        burst_rate=1., burst_std=20., numspikes=2, spike_isi=10, repeats=10,
        location='proximal', weights_ampa=weights_ampa_p1,
        synaptic_delays=syn_delays_p1, seedcore=8)

    return net

###############################################################################
# We can now use our functions to create three distinct simulations:
# 1) beta event only, 2) ERP only, and 3) beta event + ERP.
beta_start, stimulus_start = 50.0, 125.0
net_beta = net.copy()
net_beta = add_beta_drives(net_beta, beta_start)

net_erp = net.copy()
net_erp = add_erp_drives(net_erp, stimulus_start)

net_beta_erp = net_beta.copy()
net_beta_erp = add_erp_drives(net_beta_erp, stimulus_start)

###############################################################################
# And finally we simulate.
dpls_beta = simulate_dipole(net_beta, postproc=False)
dpls_erp = simulate_dipole(net_erp, postproc=False)
dpls_beta_erp = simulate_dipole(net_beta_erp, postproc=False)

###############################################################################
# By inspecting the activity during the beta event, we can see that spiking
# occurs exclusively at 50 ms, the peak of the gaussian distributed proximal
# and distal inputs. This spiking activity leads to sustained GABAb mediated
# inhibition of the L2 and L5 pyrmaidal cells.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7),
                         constrained_layout=True)
net_beta.cell_response.plot_spikes_hist(ax=axes[0], show=False)
axes[0].set_title('Beta Event Generation')
plot_dipole(dpls_beta, ax=axes[1], layer='agg', tmin=1.0, show=False)
net_beta.cell_response.plot_spikes_raster(ax=axes[2], show=False)
axes[2].set_title('Spike Raster')

###############################################################################
# By inspecting the activity during the beta event, we can see that spiking
# occurs exclusively at 50 ms, the peak of the gaussian distributed proximal
# and distal inputs. This spiking activity leads to sustained GABAb mediated
# inhibition of the L2 and L5 pyrmaidal cells.
dpls_beta_erp[0].smooth(45)
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7),
                         constrained_layout=True)
plot_dipole(dpls_beta_erp, ax=axes[0], layer='agg', tmin=1.0, show=False)
axes[0].set_title('Beta Event + ERP')
net_beta_erp.cell_response.plot_spikes_hist(ax=axes[1], show=False)
axes[1].set_title('Input Drives Histogram')
net_beta_erp.cell_response.plot_spikes_raster(ax=axes[2], show=False)
axes[2].set_title('Spike Raster')

###############################################################################
# One effect of this inhibition is an assymetric beta event with a long
# positive tail. The sustained inhibition of the network ultimately depressing
# the sensory response which is assoicated with a reduced ERP amplitude
dpls_erp[0].smooth(45)
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7),
                         constrained_layout=True)
plot_dipole(dpls_beta_erp, ax=axes[0], layer='agg', tmin=1.0, show=False)
plot_dipole(dpls_erp, ax=axes[0], layer='agg', tmin=1.0, show=False)
axes[0].set_title('Beta ERP Comparison')
axes[0].legend(['ERP + Beta', 'ERP'])
net_beta_erp.cell_response.plot_spikes_raster(ax=axes[1], show=False)
axes[1].set_title('Beta + ERP Spike Raster')
net_erp.cell_response.plot_spikes_raster(ax=axes[2], show=False)
axes[2].set_title('ERP Spike Raster')

###############################################################################
# References
# ----------
# .. [1] Law, R. G., Pugliese, S., Shin, H., Sliva, D. D., Lee, S.,
#        Neymotin, S., Moore, C., & Jones, S. R. (2021). Thalamocortical
#        mechanisms regulating the relationship between transient beta events
#        and human tactile perception. BioRxiv, 2021.04.16.440210.
#        https://doi.org/10.1101/2021.04.16.440210
