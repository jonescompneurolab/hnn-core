"""
================================================
05. From MEG sensor-space data to HNN simulation
================================================

This example demonstrates how to calculate an inverse solution of the median
nerve evoked response potential (ERP) in S1 from the MNE somatosensory dataset,
and then simulate a biophysical model network that reproduces the observed
dynamics. Note that we do not expound on how we came up with the sequence of
evoked drives used in this example, rather, we only demonstrate its
implementation. For those who want more background on the HNN model and the
process used to articulate the proximal and distal drives needed to simulate
evoked responses, see the `HNN ERP tutorial`_. The sequence of evoked drives
presented here is not part of a current publication but is motivated by prior
studies [1]_, [2]_.
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
l_freq, h_freq = 1, 40
raw.filter(l_freq, h_freq)

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
# to the cortical surface. Hence, using the option ``pick_ori='normal'`` is
# appropriate.
snr = 3.
lambda2 = 1. / snr ** 2
evoked = epochs.average()
stc = apply_inverse(evoked, inv, lambda2, method='MNE',
                    pick_ori="normal", return_residual=False,
                    verbose=True)

###############################################################################
# To extract the primary response in primary somatosensory cortex (S1), we
# create a label for the postcentral gyrus (S1) in source-space
hemi = 'rh'
label_tag = 'G_postcentral'
label_s1 = mne.read_labels_from_annot(subject, parc='aparc.a2009s', hemi=hemi,
                                      regexp=label_tag,
                                      subjects_dir=subjects_dir)[0]

###############################################################################
# Visualizing the distributed S1 activation in reference to the geometric
# structure of the cortex (i.e., plotted on a structural MRI) can help us
# figure out how to orient the dipole. Note that in the HNN framework,
# positive and negative deflections of a current dipole source correspond to
# upwards (from deep to superficial) and downwards (from superficial to deep)
# current flow, respectively. Uncomment the following code to open an
# interactive 3D render of the brain and its surface activation (requires the
# ``pyvista`` python library). You should get 2 plots, the first showing the
# post-central gyrus label from which the dipole time course was extracted and
# the second showing MNE activation at 0.040 sec that resemble the following
# images.
'''
Brain = mne.viz.get_brain_class()
brain_label = Brain(subject, hemi, 'white', subjects_dir=subjects_dir)
brain_label.add_label(label_s1, color='green', alpha=0.9)
stc_label = stc.in_label(label_s1)
brain = stc_label.plot(subjects_dir=subjects_dir, hemi=hemi, surface='white',
                       view_layout='horizontal', initial_time=0.04,
                       backend='pyvista')
'''

###############################################################################
# |mne_label_fig|
#
# |mne_activity_fig|

###############################################################################
# Now we extract the representative time course of dipole activation in our
# labeled brain region using ``mode='pca_flip'`` (see `this MNE-python
# example`_ for more details). Note that the most prominent component of the
# median nerve response occurs in the posterior wall of the central sulcus at
# ~0.040 sec. Since the dipolar activity here is negative, we orient the
# extracted waveform so that the deflection at ~0.040 sec is pointed downwards.
# Thus, the ~0.040 sec deflection corresponds to current flow traveling from
# superficial to deep layers of cortex.
flip_data = stc.extract_label_time_course(label_s1, inv['src'],
                                          mode='pca_flip')
dipole_tc = -flip_data[0] * 1e9

plt.figure()
plt.plot(1e3 * stc.times, dipole_tc, 'ro--')
plt.xlabel('Time (ms)')
plt.ylabel('Current Dipole (nAm)')
plt.xlim((0, 170))
plt.axhline(0, c='k', ls=':')
plt.show()

###############################################################################
# Now, let us try to simulate the same with ``hnn-core``. We read in the
# network parameters from ``N20.json`` and instantiate the network.

import hnn_core
from hnn_core import simulate_dipole, read_params, default_network
from hnn_core import average_dipoles, JoblibBackend

hnn_core_root = op.dirname(hnn_core.__file__)

params_fname = op.join(hnn_core_root, 'param', 'N20.json')
params = read_params(params_fname)

net = default_network(params)

###############################################################################
# To simulate the source of the median nerve evoked response, we add a
# sequence of synchronous evoked drives: 1 proximal, 2 distal, and 1 final
# proximal drive. In order to understand the physiological implications of
# proximal and distal drive as well as the general process used to articulate
# a sequence of exogenous drive for simulating evoked responses, see the
# `HNN ERP tutorial`_. Note that setting ``sync_within_trial=True`` creates
# drives with synchronous input (arriving to and transmitted by hypothetical
# granular cells at the center of the network) to all pyramidal and basket
# cells that receive distal drive. Note that granule cells are not explicitly
# modelled within HNN.

# Early proximal drive
weights_ampa_p = {'L2_basket': 0.0036, 'L2_pyramidal': 0.0039,
                  'L5_basket': 0.0019, 'L5_pyramidal': 0.0020}
weights_nmda_p = {'L2_basket': 0.0029, 'L2_pyramidal': 0.0005,
                  'L5_basket': 0.0030, 'L5_pyramidal': 0.0019}
synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_basket': 1.0, 'L5_pyramidal': 1.0}

net.add_evoked_drive(
    'evprox1', mu=21., sigma=4., numspikes=1, sync_within_trial=True,
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
    'evprox2', mu=134., sigma=4.5, numspikes=1, sync_within_trial=True,
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
    'evdist1', mu=32., sigma=2.5, numspikes=1, sync_within_trial=True,
    weights_ampa=weights_ampa_d, weights_nmda=weights_nmda_d,
    location='distal', synaptic_delays=synaptic_delays_d, seedcore=6)

# Late distal drive
weights_ampa_d = {'L2_basket': 0.0041, 'L2_pyramidal': 0.0019,
                  'L5_pyramidal': 0.0018}
weights_nmda_d = {'L2_basket': 0.0032, 'L2_pyramidal': 0.0018,
                  'L5_pyramidal': 0.0017}
synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_pyramidal': 0.1}

net.add_evoked_drive(
    'evdist2', mu=84., sigma=4.5, numspikes=1, sync_within_trial=True,
    weights_ampa=weights_ampa_d, weights_nmda=weights_nmda_d,
    location='distal', synaptic_delays=synaptic_delays_d, seedcore=2)

###############################################################################
# Now we run the simulation over 2 trials so that we can plot the average
# aggregate dipole. For a better match to the empirical waveform, set
# ``n_trials`` to be >=25.
n_trials = 2
# n_trials = 25
with JoblibBackend(n_jobs=2):
    dpls = simulate_dipole(net, n_trials=n_trials, postproc=False)

###############################################################################
# Since the model is a reduced representation of the larger network
# contributing to the response, the model response is noisier than it would be
# in the net activity from a larger network where these effects are averaged
# out, and the dipole amplitude is smaller than the recorded data. The
# post-processing steps of smoothing and scaling the simulated dipole response
# allow us to more accurately approximate the true signal responsible for the
# recorded macroscopic evoked response [1]_, [2]_.
dpl_smooth_win = 20
dpl_scalefctr = 12
for dpl in dpls:
    dpl.smooth(dpl_smooth_win)
    dpl.scale(dpl_scalefctr)

###############################################################################
# Finally, we plot the driving spike histogram, empirical and simulated median
# nerve evoked response waveforms, and output spike histogram.
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
net.cell_response.plot_spikes_hist(ax=axes[0],
                                   spike_types=['evprox', 'evdist'],
                                   show=False)
axes[1].axhline(0, c='k', ls=':', label='_nolegend_')
axes[1].plot(1e3 * stc.times, dipole_tc, 'r--')
average_dipoles(dpls).plot(ax=axes[1], show=False)
axes[1].legend(['MNE label average', 'HNN simulation'])
axes[1].set_ylabel('Current Dipole (nAm)')
net.cell_response.plot_spikes_raster(ax=axes[2])

###############################################################################
# References
# ----------
# .. [1] Jones, S. R., Pritchett, D. L., Stufflebeam, S. M., Hämäläinen, M.
#    & Moore, C. I. Neural correlates of tactile detection: a combined
#    magnetoencephalography and biophysically based computational modeling
#    study. J. Neurosci. 27, 10751–10764 (2007).
# .. [2] Neymotin SA, Daniels DS, Caldwell B, McDougal RA, Carnevale NT,
#    Jas M, Moore CI, Hines ML, Hämäläinen M, Jones SR. Human Neocortical
#    Neurosolver (HNN), a new software tool for interpreting the cellular and
#    network origin of human MEG/EEG data. eLife 9, e51214 (2020).
#    https://doi.org/10.7554/eLife.51214

###############################################################################
# .. LINKS
#
# .. _MNE: https://mne.tools/
# .. _HNN ERP tutorial: https://jonescompneurolab.github.io/hnn-tutorials/erp/erp
# .. _this MNE-python example: https://mne.tools/stable/auto_examples/inverse/plot_label_source_activations.html#sphx-glr-auto-examples-inverse-plot-label-source-activations-py
# .. |mne_label_fig| image:: https://user-images.githubusercontent.com/20212206/106524603-cfe75c80-64b0-11eb-9607-3415195c3e7a.png # noqa
#   :width: 400
# .. |mne_activity_fig| image:: https://user-images.githubusercontent.com/20212206/106524542-b514e800-64b0-11eb-835e-497454e75eb9.png # noqa
#   :width: 400
