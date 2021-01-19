"""
=================================
05. Source reconstruction and HNN
=================================

This example demonstrates how to simulate the source time
courses obtained during median nerve stimulation in the MNE
somatosensory dataset.
"""

# Authors: Mainak Jas <mainakjas@gmail.com>
#          Ryan Thorpe <ryan_thorpe@brown.edu>

###############################################################################
# First, we will import the packages and define the paths. For this example,
# we will need `MNE`_ installed. For most practical purposes, you can simply
# do:
#
#   $ pip install mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import somato
from mne.minimum_norm import apply_inverse, make_inverse_operator

data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))
fwd_fname = op.join(data_path, 'derivatives', 'sub-{}'.format(subject),
                    'sub-{}_task-{}-fwd.fif'.format(subject, task))
subjects_dir = op.join(data_path, 'derivatives', 'freesurfer', 'subjects')

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

###############################################################################
# There are several methods to do source reconstruction. Some of the methods
# such as MNE are distributed source methods whereas dipole fitting will
# estimate the location and amplitude of a single current dipole. At the
# moment, we do not offer explicit recommendations on which source
# reconstruction technique is best for HNN. However, we do want our users
# to note that the dipole currents simulate with HNN are assumed to be normal
# to the cortical surface. Hence, using the option ``pick_ori='normal'``
# seems to make most sense.

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
# Now, let us try to simulate the same with hnn-core. We read in the network
# parameters from ``N20.json`` and explicitly create two distal and one
# proximal evoked drive.

import os.path as op

import hnn_core
from hnn_core import simulate_dipole, read_params, Network

hnn_core_root = op.dirname(hnn_core.__file__)

params_fname = op.join(hnn_core_root, 'param', 'N20.json')
params = read_params(params_fname)

net = Network(params)

# Distal evoked drives share connection parameters
weights_ampa_d = {'L2_basket': 0.003, 'L2_pyramidal': 0.0045,
                  'L5_pyramidal': 0.001}
weights_nmda_d = {'L2_basket': 0.003, 'L2_pyramidal': 0.0045,
                  'L5_pyramidal': 0.001}
synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_pyramidal': 0.1}
# early distal input
net.add_evoked_drive(
    'evdist1', mu=32., sigma=0., numspikes=1, weights_ampa=weights_ampa_d,
    weights_nmda=weights_nmda_d, location='distal',
    synaptic_delays=synaptic_delays_d, seedcore=6)
# late distal input
net.add_evoked_drive(
    'evdist2', mu=82., sigma=0., numspikes=1, weights_ampa=weights_ampa_d,
    weights_nmda=weights_nmda_d, location='distal',
    synaptic_delays=synaptic_delays_d, seedcore=2)

# proximal input occurs before distals
weights_ampa_p = {'L2_basket': 0.003, 'L2_pyramidal': 0.0025,
                  'L5_basket': 0.004, 'L5_pyramidal': 0.001}
weights_nmda_p = {'L2_basket': 0.003, 'L5_basket': 0.004}
synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_basket': 1.0, 'L5_pyramidal': 1.0}
net.add_evoked_drive(
    'evprox1', mu=20.0, sigma=0., numspikes=1, weights_ampa=weights_ampa_p,
    weights_nmda=weights_nmda_p, location='proximal',
    synaptic_delays=synaptic_delays_p, seedcore=6)

dpl = simulate_dipole(net, n_trials=1)

trial_idx = 0
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
dpl[trial_idx].plot(ax=axes[0], show=False)
net.cell_response.plot_spikes_hist(ax=axes[1])
net.cell_response.plot_spikes_raster()

###############################################################################
# .. LINKS
#
# .. _MNE: https://mne.tools/
