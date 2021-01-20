"""
========================
02. Simulate alpha waves
========================

This example demonstrates how to simulate alpha waves using
HNN-core.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network

###############################################################################
# Then we setup the directories and read the default parameters file
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

weights_ampa = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
net.add_bursty_drive(
    'bursty', tstart=50., burst_rate=10, burst_std=20., numspikes=2,
    spike_isi=10, repeats=10, location='distal', weights_ampa=weights_ampa,
    seedcore=4)

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
from scipy.signal import spectrogram
import numpy as np

###############################################################################
# Next we define sfreq, the sampling frequency of the simulation. This can be
# calculated from the step size defined used for the differential equation
# solver. We additionally  define n_fft, the length of the fast fourier
# transform.
sfreq = 1000. / params['dt']
n_fft = 1024 * 8
freqs, _, psds = spectrogram(
    dpl[0].data['agg'], sfreq, window='hamming', nfft=n_fft,
    nperseg=n_fft, noverlap=0)

###############################################################################
# Finally we plot the PSD and confirm the presence of alpha activity from the
# sharp peak centered at 10 Hz.
plt.figure()
plt.plot(freqs, np.mean(psds, axis=-1))
plt.xlim((0, 40))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.show()
