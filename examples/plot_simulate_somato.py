"""
====================
Simulate somato data
====================

This example demonstrates how to simulate the source time
courses obtained during median nerve stimulation in the MNE
somatosensory dataset.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

###############################################################################
# First, we will import the packages and define the paths

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import somato
from mne.minimum_norm import apply_inverse, make_inverse_operator

data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
fwd_fname = data_path + '/MEG/somato/somato-meg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'

###############################################################################
# Then, we get the raw data and estimage the source time course

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40)

events = mne.find_events(raw, stim_channel='STI 014')
event_id, tmin, tmax = 1, -.2, .15
baseline = None
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                    reject=dict(grad=4000e-13, eog=350e-6), preload=True)
evoked = epochs.average()

fwd = mne.read_forward_solution(fwd_fname)
cov = mne.compute_covariance(epochs)
inv = make_inverse_operator(epochs.info, fwd, cov)

method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inv, lambda2, method=method, pick_ori="normal",
                    return_residual=False, verbose=True)

pick_vertex = np.argmax(np.linalg.norm(stc.data, axis=1))

plt.figure()
plt.plot(1e3 * stc.times, stc.data[pick_vertex, :].T * 1e9, 'ro-')
plt.xlabel('time (ms)')
plt.ylabel('%s value (nAM)' % method)
plt.xlim((0, 150))
plt.axhline(0)
plt.show()

###############################################################################
# Now, let us try to simulate the same with MNE-neuron

import os.path as op

import mne_neuron
from mne_neuron import simulate_dipole, Params, Network

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

params_fname = op.join(mne_neuron_root, 'param', 'N20.json')
params = Params(params_fname)

net = Network(params)
dpl = simulate_dipole(net)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
dpl.plot(ax=axes[0])
net.plot_input(ax=axes[1])
net.plot_spikes()
