"""
=================================
05. Source reconstruction and HNN
=================================

This example demonstrates how to calculate the inverse solution of the median
nerve evoked response in the MNE somatosensory dataset, and then simulate a
matching inverse solution with HNN.
"""

# Authors: Mainak Jas <mainakjas@gmail.com>
#          Ryan Thorpe <ryan_thorpe@brown.edu>

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# First, we will import the packages needed for computing the inverse solution
# from the MNE somatosensory dataset. `MNE`_ can be installed with
# ``pip install mne``, and the somatosensory dataset can be downloaded by
# importing ``somato`` from ``mne.datasets``.
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import somato
from mne.minimum_norm import apply_inverse, make_inverse_operator

###############################################################################
# Now we set the the path of the ``somato`` dataset for subject ``'01'``.
data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))
fwd_fname = op.join(data_path, 'derivatives', 'sub-{}'.format(subject),
                    'sub-{}_task-{}-fwd.fif'.format(subject, task))
subjects_dir = op.join(data_path, 'derivatives', 'freesurfer', 'subjects')

###############################################################################
# Then, we load the raw data and estimate the inverse operator.

# Read and band-pass filter the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40)

# Identify stimulus events associated with MEG time series in the dataset
events = mne.find_events(raw, stim_channel='STI 014')

# Define epochs within the time series
event_id, tmin, tmax = 1, -.2, .17
baseline = None
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                    reject=dict(grad=4000e-13, eog=350e-6), preload=True)

# Compute the inverse operator
fwd = mne.read_forward_solution(fwd_fname)
cov = mne.compute_covariance(epochs)
inv = make_inverse_operator(epochs.info, fwd, cov)

###############################################################################
# There are several methods to do source reconstruction. Some of the methods
# such as MNE are distributed source methods whereas dipole fitting will
# estimate the location and amplitude of a single current dipole. At the
# moment, we do not offer explicit recommendations on which source
# reconstruction technique is best for HNN. However, we do want our users
# to note that the dipole currents simulated with HNN are assumed to be normal
# to the cortical surface. Hence, using the option ``pick_ori='normal'``
# seems to make most sense.
method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
evoked = epochs.average()
stc = apply_inverse(evoked, inv, lambda2, method=method, pick_ori="normal",
                    return_residual=False, verbose=True)

###############################################################################
# We isolate and plot the single most active vertex in the distributed minimum
# norm estimate by calculating the L2 norm of the time course emerging from
# each vertex. The time course from the vertex with the greatest L2 norm
# represents the location of cortex with greatest response to stimulus.
pick_vertex = np.argmax(np.linalg.norm(stc.data, axis=1))

plt.figure()
plt.plot(1e3 * stc.times, stc.data[pick_vertex, :].T * 1e9, 'ro--')
plt.xlabel('Time (ms)')
plt.ylabel('Current Dipole (nAm)')
plt.xlim((0, 170))
plt.axhline(0, c='k', ls=':')
plt.show()

###############################################################################
# Now, let us try to simulate the same with hnn-core. We read in the network
# parameters from ``N20.json`` and instantiate the network.

import hnn_core
from hnn_core import simulate_dipole, read_params, Network, MPIBackend, average_dipoles

hnn_core_root = op.dirname(hnn_core.__file__)

params_fname = op.join(hnn_core_root, 'param', 'N20.json')
params = read_params(params_fname)

net = Network(params)

###############################################################################
# To simulate the source of the median nerve evoked response, we add a
# sequence of synchronous evoked drives: 1 proximal, 2 distal, and 1 final
# proximal drive. Note that setting ``sync_within_trial=True`` creates drives
# with synchronous input (arriving to and transmitted by hypothetical granular
# cells at the center of the network) to all pyramidal and basket cells that
# receive distal drive.

# Early proximal drive
weights_ampa_p = {'L2_basket': 0.0036, 'L2_pyramidal': 0.0039,
                  'L5_basket': 0.0019, 'L5_pyramidal': 0.0020}
weights_nmda_p = {'L2_basket': 0.0029, 'L2_pyramidal': 0.0005,
                  'L5_basket': 0.0030, 'L5_pyramidal': 0.0019}
synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_basket': 1.0, 'L5_pyramidal': 1.0}

net.add_evoked_drive(
    'evprox1', mu=20.0, sigma=3., numspikes=1, sync_within_trial=True,
    weights_ampa=weights_ampa_p, weights_nmda=weights_nmda_p,
    location='proximal', synaptic_delays=synaptic_delays_p, seedcore=6)

# Late proximal drive
weights_ampa_p = {'L2_basket': 0.003, 'L2_pyramidal': 0.0039,
                  'L5_basket': 0.004, 'L5_pyramidal': 0.0020}
weights_nmda_p = {'L2_basket': 0.001, 'L2_pyramidal': 0.0005,
                  'L5_basket': 0.002, 'L5_pyramidal': 0.0020}
synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_basket': 1.0, 'L5_pyramidal': 1.0}

net.add_evoked_drive(
    'evprox2', mu=130.0, sigma=3., numspikes=1, sync_within_trial=True,
    weights_ampa=weights_ampa_p, weights_nmda=weights_nmda_p,
    location='proximal', synaptic_delays=synaptic_delays_p, seedcore=6)

# Early distal drive
weights_ampa_d = {'L2_basket': 0.0043, 'L2_pyramidal': 0.0032,
                  'L5_pyramidal': 0.0009}
weights_nmda_d = {'L2_basket': 0.0029, 'L2_pyramidal': 0.0051,
                  'L5_pyramidal': 0.0010}
synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_pyramidal': 0.1}

net.add_evoked_drive(
    'evdist1', mu=32., sigma=3., numspikes=1, sync_within_trial=True,
    weights_ampa=weights_ampa_d, weights_nmda=weights_nmda_d,
    location='distal', synaptic_delays=synaptic_delays_d, seedcore=6)

# Late distal input
weights_ampa_d = {'L2_basket': 0.0041, 'L2_pyramidal': 0.0019,
                  'L5_pyramidal': 0.0018}
weights_nmda_d = {'L2_basket': 0.0032, 'L2_pyramidal': 0.0018,
                  'L5_pyramidal': 0.0017}
synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_pyramidal': 0.1}

net.add_evoked_drive(
    'evdist2', mu=82., sigma=3., numspikes=1, sync_within_trial=True,
    weights_ampa=weights_ampa_d, weights_nmda=weights_nmda_d,
    location='distal', synaptic_delays=synaptic_delays_d, seedcore=2)

###############################################################################
# Now we run the simulation over 2 trials so that we can plot the average
# aggregate dipole. As in :ref:`the MPIBackend example
# <sphx_glr_auto_examples_plot_simulate_mpi_backend.py>`, we can use
# ``MPIBackend`` to reduce the simulation time by parallizing across cells in
# the network. However, no parallel backend is necessary. For a better
# match to the empirical waveform, set ``n_trials`` to be >=25.
n_trials = 2
# n_trials = 25
with MPIBackend(n_procs=6, mpi_cmd='mpiexec'):
    dpls = simulate_dipole(net, n_trials=n_trials)

###############################################################################
# Finally, we plot the driving spike histogram, empirical and simulated median
# nerve evoked response waveforms, and output spike histogram.
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
net.cell_response.plot_spikes_hist(ax=axes[0], show=False)
axes[0].legend(['evdist_1', 'evdist_2', 'evprox_1', 'evprox_2'])
axes[0].set_ylabel('Counts')
axes[1].axhline(0, c='k', ls=':')
axes[1].plot(1e3 * stc.times, stc.data[pick_vertex, :].T * 1e9, 'r--')
axes[1].set_ylabel('Current Dipole (nAm)')
average_dipoles(dpls).plot(ax=axes[1], show=False)
net.cell_response.plot_spikes_raster(ax=axes[2])

###############################################################################
# .. LINKS
#
# .. _MNE: https://mne.tools/
