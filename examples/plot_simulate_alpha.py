"""
====================
Simulate alpha waves
====================

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
# Then we setup the directories and Neuron
hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the default parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
print(params)

###############################################################################
# Update a few of the default parameters related to visualisation
params.update({
    'dipole_scalefctr': 150000.0,
    'dipole_smooth_win': 0,
    'sync_evinput': 1,  # XXX not applied (seed hack, see feed.py:_get_prng)
    'tstop': 310.0,
})

###############################################################################
# Now let's simulate the dipole and plot it
net = Network(params)

# XXX to match online docs, remove before MRG
net.add_bursty_drive(
    'bogus', distribution='normal', t0=50., sigma_t0=0., T=params['tstop'],
    burst_f=10, spike_jitter_std=20., numspikes=2, spike_isi=10, repeats=10,
    weights_ampa=None, weights_nmda=None, location='proximal',
    seedcore=3, space_constant=100.)

weights_ampa = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
net.add_bursty_drive(
    'bursty', distribution='normal', t0=50., sigma_t0=0., T=params['tstop'],
    burst_f=10, spike_jitter_std=20., numspikes=2, spike_isi=10, repeats=10,
    weights_ampa=weights_ampa, weights_nmda=None, location='distal',
    seedcore=3, space_constant=100.)

dpl = simulate_dipole(net, n_trials=1)  # XXX n_trials=1 instantiates drive!
dpl[0].plot()

###############################################################################
# We can confirm that what we simulate is indeed 10 Hz activity.
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
sfreq = 1000. / params['dt']
n_fft = 1024 * 8
freqs, _, psds = spectrogram(
    dpl[0].data['agg'], sfreq, window='hamming', nfft=n_fft,
    nperseg=n_fft, noverlap=0)
plt.figure()
plt.plot(freqs, np.mean(psds, axis=-1))
plt.xlim((0, 40))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.show()
