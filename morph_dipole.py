import os.path as op
import tempfile

import matplotlib.pyplot as plt
import hnn_core
from hnn_core import simulate_dipole
from hnn_core.viz import plot_dipole
from hnn_core.network_models import random_model, custom_cell_types_model

# Test 1: Using random_model with L2_random instead of L2_basket
print("=== Testing with random_model (L2_random cells) ===")
net = random_model(mesh_shape=(5, 5))  # Larger network for better visualization
net.plot_cells()

# Check if we can plot morphology of our cells
if 'L5_pyramidal' in net.cell_types:
    net.cell_types['L5_pyramidal'].plot_morphology()

# Update weights to include L2_random instead of L2_basket
weights_ampa_d1 = {'L2_random': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_random': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_random': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}

net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=274)

weights_ampa_p1 = {'L2_random': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_random': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=544)

weights_ampa_p2 = {'L2_random': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}

net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=814)

# Simulate with the random model
from hnn_core import JoblibBackend

with JoblibBackend(n_jobs=2):
    dpls = simulate_dipole(net, tstop=170., n_trials=2)

window_len, scaling_factor = 30, 3000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

# Plot results
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1],
                                   spike_types=['evprox', 'evdist'])
plt.suptitle('Random Model with L2_random cells')

# Test layer plots
plot_dipole(dpls, average=False, layer=['L2', 'L5', 'agg'], show=False)

# Test 2: Using custom_cell_types_model with completely new cell types
print("\n=== Testing with custom_cell_types_model ===")
net_custom = custom_cell_types_model(mesh_shape=(4, 4))
net_custom.plot_cells()

# Define weights for our custom cell types
weights_ampa_d1_custom = {
    'L2_interneuron': 0.006562,  # Similar to L2_basket
    'L2_pyramidal': .000007,
    'L4_stellate': 0.05,          # New intermediate layer
    'L5_pyramidal': 0.142300
}
weights_nmda_d1_custom = {
    'L2_interneuron': 0.019482,
    'L2_pyramidal': 0.004317,
    'L4_stellate': 0.03,
    'L5_pyramidal': 0.080074
}
synaptic_delays_d1_custom = {
    'L2_interneuron': 0.1,
    'L2_pyramidal': 0.1,
    'L4_stellate': 0.5,  # Intermediate delay
    'L5_pyramidal': 0.1
}

# Add drives to custom model
net_custom.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1,
    weights_ampa=weights_ampa_d1_custom,
    weights_nmda=weights_nmda_d1_custom, location='distal',
    synaptic_delays=synaptic_delays_d1_custom, event_seed=274)

# Proximal drive weights for custom cells
weights_ampa_p1_custom = {
    'L2_interneuron': 0.08831,
    'L2_pyramidal': 0.01525,
    'L4_stellate': 0.05,
    'L5_pyramidal': 0.00865
}
synaptic_delays_prox_custom = {
    'L2_interneuron': 0.1,
    'L2_pyramidal': 0.1,
    'L4_stellate': 0.5,
    'L5_pyramidal': 1.
}

net_custom.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1,
    weights_ampa=weights_ampa_p1_custom,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox_custom, event_seed=544)

# Simulate custom model
dpls_custom = simulate_dipole(net_custom, tstop=170., n_trials=1)

# Plot custom model results
dpls_custom[0].copy().smooth(window_len).scale(scaling_factor).plot()
plt.title('Custom Cell Types Model Dipole')

# Test synchronized drives with custom model
print("\n=== Testing synchronized drives with custom cells ===")
net_sync = random_model(mesh_shape=(5, 5))

n_drive_cells = 1
cell_specific = False

# Add synchronized drives with L2_random
net_sync.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1,
    weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
    location='distal', n_drive_cells=n_drive_cells,
    cell_specific=cell_specific, synaptic_delays=synaptic_delays_d1,
    event_seed=274)

net_sync.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1,
    weights_ampa=weights_ampa_p1, weights_nmda=None,
    location='proximal', n_drive_cells=n_drive_cells,
    cell_specific=cell_specific, synaptic_delays=synaptic_delays_prox,
    event_seed=544)

net_sync.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    n_drive_cells=n_drive_cells, cell_specific=cell_specific,
    synaptic_delays=synaptic_delays_prox, event_seed=814)

print("Drive dynamics:", net_sync.external_drives['evdist1']['dynamics'])

dpls_sync = simulate_dipole(net_sync, tstop=170., n_trials=1)

trial_idx = 0
dpls_sync[trial_idx].copy().smooth(window_len).scale(scaling_factor).plot()
plt.title('Synchronized Drives with L2_random cells')

# Plot spike histogram to see our custom cells
net_sync.cell_response.plot_spikes_hist()

# Test visualization with custom colors
print("\n=== Testing custom visualization ===")
custom_colors = {
    'L2_random': 'green',
    'L2_pyramidal': 'lightblue',
    'L5_basket': 'red',
    'L5_pyramidal': 'darkblue'
}

custom_markers = {
    'L2_random': 'd',  # diamond
    'L2_pyramidal': 'v',
    'L5_basket': 'x',
    'L5_pyramidal': '^'
}

net.plot_cells(cell_colors=custom_colors, cell_markers=custom_markers)
plt.title('Network with Custom Colors and Markers')

plt.show()