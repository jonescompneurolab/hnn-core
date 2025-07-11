"""
Example script demonstrating communication between two HNN networks.

This script shows how to:
1. Create a source network (Network A) with the jones_2009_model
2. Simulate Network A and extract spike data
3. Create a target network (Network B)
4. Feed spike data from Network A to Network B using add_spike_train_drive
5. Simulate Network B and visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt
from hnn_core import read_params, Network, simulate_dipole, jones_2009_model, read_spikes
import os.path as op
import tempfile

# Create and simulate Network A (source network)
print("Creating and simulating Network A...")
net_A = jones_2009_model()
net_A._params.update({'tstop': 170.0})

# Add a drive to Network A to generate some activity
print("Adding drive to Network A...")
drive_params = {
    'name': 'evdist1', 'mu': 63.5, 'sigma': 3.8, 'numspikes': 1,
    'location': 'distal', 'event_seed': 274
}
weights = {
    'ampa': {'L2_basket': 0.006, 'L2_pyramidal': 0.0005, 'L5_pyramidal': 0.14},
    'nmda': {'L2_basket': 0.019, 'L2_pyramidal': 0.004, 'L5_pyramidal': 0.08}
}
delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}
net_A.add_evoked_drive(
    **drive_params, weights_ampa=weights['ampa'], weights_nmda=weights['nmda'],
    synaptic_delays=delays
)

# Simulate and save spike data to temp file
print("Simulating Network A and saving spikes...")
dpls_A = simulate_dipole(net_A, tstop=170.0, n_trials=1)
with tempfile.TemporaryDirectory() as tmp_dir:
    spike_file = op.join(tmp_dir, 'spk_%d.txt')
    net_A.cell_response.write(spike_file)
    cell_response = read_spikes(op.join(tmp_dir, 'spk_*.txt'))
    print(f"Extracted {sum(len(t) for t in cell_response.spike_times)} spikes from Network A")

# Create and configure Network B (target network)
print("Creating and configuring Network B...")
net_B = jones_2009_model()
net_B._params.update({'tstop': 225.0})

# Extract spike data from first trial and filter for pyramidal cells
trial_idx = 0
spike_times = cell_response.spike_times[trial_idx]
spike_gids = cell_response.spike_gids[trial_idx]
spike_types = cell_response.spike_types[trial_idx]
pyramidal_mask = np.array([t in ['L2_pyramidal', 'L5_pyramidal'] for t in spike_types])
filtered_times = np.array(spike_times)[pyramidal_mask]
filtered_gids = np.array(spike_gids)[pyramidal_mask]
filtered_types = np.array(spike_types)[pyramidal_mask]
spike_data = {f"NetA_{t}_GID{g}": [] for t, g in zip(filtered_types, filtered_gids)}
for t, g, time in zip(filtered_types, filtered_gids, filtered_times):
    src_id = f"NetA_{t}_GID{g}"
    spike_data[src_id].append(time)

# Define target configuration
conn_properties = {
    'L5_pyramidal': {'location': 'distal', 'weights_ampa': 0.005, 'synaptic_delays': 1.5},
    'L2_pyramidal': {'location': 'proximal', 'weights_ampa': 0.003, 'synaptic_delays': 1.0}
}
target_config = {}
source_ids = list(spike_data.keys())
for i, src_id in enumerate(source_ids):
    target_type = 'L5_pyramidal' if i % 2 == 0 else 'L2_pyramidal'
    props = conn_properties[target_type]
    target_config[src_id] = {
        'target_cell_types': [target_type],
        'location': props['location'],
        'weights_ampa': {target_type: props['weights_ampa']},
        'synaptic_delays': {target_type: props['synaptic_delays']},
        'probability': 0.7
    }
all_weights_ampa = {k: cfg['weights_ampa'][k] for cfg in target_config.values() for k in cfg['weights_ampa']}

# Add spike train drive to Network B
net_B.add_spike_train_drive(
    name='drive_from_NetA',
    spike_data=spike_data,
    location='distal',
    weights_ampa=all_weights_ampa,
    weights_nmda=None,
    synaptic_delays=0.1,
    conn_seed=42
)
# Simulate Network B
print("Simulating Network B...")
dpls_b = simulate_dipole(net_B, tstop=225.0, n_trials=1)

# Visualize results
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
# 1. Network A: Spike raster
net_A.cell_response.plot_spikes_raster(ax=axes[0], show=False)
axes[0].set_title('Network A: Spike Raster')
# 2. Network B: Spike raster
axes[0].set_ylabel('Cell ID')
net_B.cell_response.plot_spikes_raster(ax=axes[1], show=False)
axes[1].set_title('Network B: Spike Raster')
axes[1].set_ylabel('Cell ID')
axes[1].set_xlabel('Time (ms)')
plt.show()

# Verify drive setup
print("\nVerifying configuration:")
drive_name = 'drive_from_NetA'
if drive_name in net_B.external_drives:
    drive = net_B.external_drives[drive_name]
    print(f"  Drive '{drive_name}' created with {drive['n_drive_cells']} cells")
    if drive_name in net_B.gid_ranges:
        print(f"  GIDs assigned: {net_B.gid_ranges[drive_name]}")
    drive_connections = sum(1 for conn in net_B.connectivity if conn['src_type'] == drive_name)
    print(f"  Number of connections from drive: {drive_connections}")
else:
    print(f"  ERROR: Drive '{drive_name}' not found")