"""
=================================
02. Simulate alpha and beta waves
=================================

This example demonstrates how to simulate alpha and beta waves using
HNN-core.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network

###############################################################################
# Then we setup the directories and read the default parameters file
hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
print(params)

###############################################################################
# Next we update a few of the default parameters related to visualisation. The
# dipole_scalefctr relates to the amount of cortical tissue necessary to
# observe the electric current dipole outside the head with M/EEG.
params.update({
    'dipole_scalefctr': 150000.0,
    'dipole_smooth_win': 0,
    'tstop': 310.0,
})

###############################################################################
# Now let's simulate the dipole and plot it. To excite the network, we add a
# ~10 Hz "bursty" drive starting at 50 ms and continuing to the end of the
# simulation. Each burst consists of a pair (2) of spikes, spaced 10 ms apart.
# The occurrence of each burst is jittered by a random, normally distributed
# amount (20 ms standard deviation). We repeat the burst train 10 times, each
# time with unique randomization. The drive is only connected to the distal
# (dendritic) AMPA synapses on L2/3 and L5 pyramidal neurons.
net = Network(params)

location = 'distal'
burst_std = 20
weights_ampa_d = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
syn_delays_d = {'L2_basket': 5., 'L2_pyramidal': 5.,
                'L5_basket': 5., 'L5_pyramidal': 5.}
net.add_bursty_drive(
    'alpha_dist', tstart=50., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, repeats=10, location=location, weights_ampa=weights_ampa_d,
    synaptic_delays=syn_delays_d, seedcore=14)

dpl = simulate_dipole(net)

trial_idx = 0  # single trial simulated
dpl[trial_idx].plot()

###############################################################################
# We can confirm that what we simulate is indeed 10 Hz activity by plotting the
# power spectral density. First we'll import the spectrogram class from scipy.
import matplotlib.pyplot as plt
from hnn_core.viz import plot_spectrogram
tmin = 20  # exclude initial burn-in period
plot_spectrogram(dpl[trial_idx], fmin=0., fmax=40., tmin=tmin)

###############################################################################
# The next step is to add a simultaneous 10 Hz proximal drive. Due to the
# stochasticity of input spike timing, the proximal and distal spikes
# occasionally arrive at the same time which will result in a beta frequency
# (15-30 Hz) event.
location = 'proximal'
burst_std = 20
weights_ampa_p = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
syn_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                'L5_basket': 1., 'L5_pyramidal': 1.}

net.add_bursty_drive(
    'alpha_prox', tstart=50., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, repeats=10, location=location, weights_ampa=weights_ampa_p,
    synaptic_delays=syn_delays_p, seedcore=13)

dpl = simulate_dipole(net)

###############################################################################
# It can be difficult to identify beta activity by inspecting the dipole
# directly. One useful tool is to plot the time frequency spectrogram.
# Create an fixed-step tiling of frequencies from 20 to 100 Hz in steps of 1 Hz
from hnn_core.viz import plot_dipole, plot_tfr_morlet
import numpy as np
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

plot_dipole(dpl[trial_idx], ax=axes[0], show=False)


freqs = np.arange(2., 50., 1.)
plot_tfr_morlet(dpl[trial_idx], freqs=freqs, n_cycles=7, ax=axes[1])