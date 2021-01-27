"""
=================================
02. Simulate alpha and beta waves
=================================

This example demonstrates how to simulate alpha and beta waves using
HNN-core. Alpha activity can be produced with 10 Hz excitatory drive to the
proximal or distal dendrites of pyramidal neurons. Providing proximal and
distal drive simultaneously results in higher frequency beta activity [1]_,
[2]_.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Christopher Bailey <bailey.cj@gmail.com>

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

###############################################################################
# Now let's simulate the dipole and plot it. To excite the network, we add a
# ~10 Hz "bursty" drive starting at 50 ms and continuing to the end of the
# simulation. Each burst consists of a pair (2) of spikes, spaced 10 ms apart.
# The occurrence of each burst is jittered by a random, normally distributed
# amount (20 ms standard deviation). We repeat the burst train 10 times, each
# time with unique randomization. The drive is only connected to the proximal
# (dendritic) AMPA synapses on L2/3 and L5 pyramidal neurons.
params['tstop'] = 310
net = Network(params)

location = 'proximal'
burst_std = 20
weights_ampa_p = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
syn_delays_p = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}

net.add_bursty_drive(
    'alpha_prox', tstart=50., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, repeats=10, location=location, weights_ampa=weights_ampa_p,
    synaptic_delays=syn_delays_p, seedcore=14)

dpl = simulate_dipole(net, postproc=False)

###############################################################################
# We can confirm that what we simulate is indeed 10 Hz activity by plotting the
# power spectral density (PSD). Note that the SciPy-function
# `~scipy.signal.periodogram` is used to create the plot. Prior to
# plotting, we smooth the dipole waveform with a _Satizky-Golay_ filter from
# SciPy (`~scipy.signal.savgol_filter`). The filter takes a single argument:
# the approximate low-pass cut-off frequency in Hz. Note that the purpose of
# the filter is here to simulate the physiological scenario in which a larger
# number and greater volume of neurons generate extra-cranially measured
# signals. Try running the code without smoothing to compare with the raw
# simulation output!
import matplotlib.pyplot as plt
from hnn_core.viz import plot_dipole, plot_periodogram
trial_idx = 0  # single trial simulated
fig, axes = plt.subplots(2, 1)
tmin = 20  # exclude initial burn-in period
h_freq = 30  # highest frequency (in Hz) to retain after smoothing (approx.)

# We'll make a copy of the dipole before smoothing
smooth_dpl = dpl[trial_idx].copy().savgol_filter(h_freq)
plot_dipole(smooth_dpl, tmin=tmin, ax=axes[0], show=False)

plot_periodogram(dpl[trial_idx], fmin=0., fmax=40., tmin=tmin, ax=axes[1])
plt.tight_layout()
###############################################################################
# The next step is to add a simultaneous 10 Hz distal drive with a lower
# within-burst spread of spike times (``burst_std``) compared with the
# proximal one. The different arrival times of spikes at opposite ends of
# the pyramidal cells will tend to produce bursts of 15-30 Hz power known
# as beta frequency events.
location = 'distal'
burst_std = 15
weights_ampa_d = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
syn_delays_d = {'L2_pyramidal': 5., 'L5_pyramidal': 5.}
net.add_bursty_drive(
    'alpha_dist', tstart=50., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, repeats=10, location=location, weights_ampa=weights_ampa_d,
    synaptic_delays=syn_delays_d, seedcore=16)

dpl = simulate_dipole(net, postproc=False)

###############################################################################
# We can verify that beta frequency activity was produced by inspecting the PSD
# of the most recent simulation. The dominant power in the signal is shifted
# from alpha (~10 Hz) to beta (15-25 Hz) frequency range.
trial_idx, tmin, h_freq = 0, 20, 30  # same as above
fig, axes = plt.subplots(2, 1)
# We'll again make a copy of the dipole before smoothing
smooth_dpl = dpl[trial_idx].copy().savgol_filter(h_freq)

# Note that using the plot_dipole-function is equivalent to:
smooth_dpl.plot(tmin=tmin, ax=axes[0], show=False)

plot_periodogram(dpl[trial_idx], fmin=0., fmax=40., tmin=tmin, ax=axes[1])
plt.tight_layout()

###############################################################################
# References
# ----------
# .. [1] Jones, S. R. et al. Quantitative analysis and biophysically realistic
#    neural modeling of the MEG mu rhythm: rhythmogenesis and modulation of
#    sensory-evoked responses. J. Neurophysiol. 102, 3554–3572 (2009).
#
# .. [2] https://jonescompneurolab.github.io/hnn-tutorials/alpha_and_beta/alpha_and_beta
