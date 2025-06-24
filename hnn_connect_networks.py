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
import hnn_core
from hnn_core import read_params, Network, simulate_dipole, jones_2009_model
import os.path as op
import csv

# Create and simulate Network A (source network)
print("Creating Network A...")
net_A = jones_2009_model()
net_A._params.update({'tstop': 170.0})

# Add a drive to Network A to generate some activity
print("Adding drive to Network A...")
weights_ampa = {'L2_basket': 0.006, 'L2_pyramidal': 0.0005, 'L5_pyramidal': 0.14}
weights_nmda = {'L2_basket': 0.019, 'L2_pyramidal': 0.004, 'L5_pyramidal': 0.08}
synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}
net_A.add_evoked_drive(
    'evdist1', 
    mu=63.5, 
    sigma=3.8, 
    numspikes=1,
    weights_ampa=weights_ampa,
    weights_nmda=weights_nmda,
    location='distal',
    synaptic_delays=synaptic_delays,
    event_seed=274
)

print("Simulating Network A...")
dpls_A = simulate_dipole(net_A, tstop=170.0, n_trials=1)

# Extract spike data from Network A
print("Extracting spike data from Network A...")
spike_data = {}

if net_A.cell_response and net_A.cell_response.spike_times:
    # Get spike data from the first trial
    trial_idx = 0
    spike_times = net_A.cell_response.spike_times[trial_idx]
    spike_gids = net_A.cell_response.spike_gids[trial_idx]
    spike_types = net_A.cell_response.spike_types[trial_idx]
    
    # Keep only pyramidal cells (as an example filter)
    pyramidal_mask = np.array([cell_type in ['L2_pyramidal', 'L5_pyramidal'] 
                              for cell_type in spike_types])
    
    filtered_times = np.array(spike_times)[pyramidal_mask]
    filtered_gids = np.array(spike_gids)[pyramidal_mask]
    filtered_types = np.array(spike_types)[pyramidal_mask]
    
    # Create source identifiers
    for i in range(len(filtered_times)):
        gid = filtered_gids[i]
        cell_type = filtered_types[i]
        time = filtered_times[i]
        src_id = f"NetA_{cell_type}_GID{gid}"
        
        if src_id not in spike_data:
            spike_data[src_id] = []
        
        spike_data[src_id].append(time)

# Print spike data summary
total_spikes = sum(len(spikes) for spikes in spike_data.values())
print(f"Extracted {total_spikes} spikes from {len(spike_data)} unique source cells")

# Create Network B (target network)
print("Creating Network B...")
net_B = jones_2009_model()
net_B._params.update({'tstop': 225.0})


# Define target configuration
target_config = {}
conn_properties = {
    'L5_pyramidal': {
        'location': 'distal',
        'weights_ampa': 0.005,
        'synaptic_delays': 1.5,
    },
    'L2_pyramidal': {
        'location': 'proximal',
        'weights_ampa': 0.003,
        'synaptic_delays': 1.0,
    }
}

# Assign half of source cells to target L5 and half to target L2
source_ids = list(spike_data.keys())
for i, src_id in enumerate(source_ids):
    # Alternate between targeting L5 and L2 pyramidal cells
    if i % 2 == 0:
        target_cell_type = 'L5_pyramidal'
    else:
        target_cell_type = 'L2_pyramidal'
        
    props = conn_properties[target_cell_type]
    
    # Create configuration for this source
    target_config[src_id] = {
        'target_cell_types': [target_cell_type],
        'location': props['location'],
        'weights_ampa': {target_cell_type: props['weights_ampa']},
        'synaptic_delays': {target_cell_type: props['synaptic_delays']},
        'probability': 0.7  # 70% connection probability as an example
    }
all_weights_ampa = {}
for cfg in target_config.values():
    all_weights_ampa.update(cfg['weights_ampa'])

net_B.add_spike_train_drive(
    name='drive_from_NetA',
    spike_data=spike_data,
    location='distal',
    weights_ampa=all_weights_ampa,  # Now has actual weights
    weights_nmda=None,
    synaptic_delays=0.1,
    conn_seed=42
)
    
    # Alternative Format 2 approach (times/gids format)
    # First, convert to times/gids format
# '''
# all_times = []
# all_gids = []
# gid_map = {}

# for i, src_id in enumerate(spike_data.keys()):
#     gid_map[src_id] = i
#     times = spike_data[src_id]
#     all_times.extend(times)
#     all_gids.extend([i] * len(times))

# times_gids_format = {
#     'times': all_times,
#     'gids': all_gids
# }

# # Demonstrate how to use Format 2
# net_B.add_spike_train_drive(
#     name='drive_from_NetA',
#     spike_data=times_gids_format,
#     location='distal',
#     weights_ampa={'L5_pyramidal': 0.005, 'L2_pyramidal': 0.003},
#     weights_nmda=None,
#     synaptic_delays={'L5_pyramidal': 1.5, 'L2_pyramidal': 1.0},
#     probability=0.7,
#     conn_seed=42
# )
# '''

# Verify drive setup
print("\nVerifying configuration:")
drive_name = 'drive_from_NetA'
if drive_name in net_B.external_drives:
    drive = net_B.external_drives[drive_name]
    print(f"  Drive '{drive_name}' successfully created")
    print(f"  Type: {drive['type']}")
    print(f"  Number of drive cells: {drive['n_drive_cells']}")
    
    # Check GID ranges
    if drive_name in net_B.gid_ranges:
        print(f"  GIDs assigned: {net_B.gid_ranges[drive_name]}")
    else:
        print("  ERROR: No GIDs assigned to drive")
    
    # Check connections
    drive_connections = 0
    for conn in net_B.connectivity:
        if conn['src_type'] == drive_name:
            drive_connections += 1
    print(f"  Number of connections from drive: {drive_connections}")
else:
    print(f"  ERROR: Drive '{drive_name}' not found in external_drives")

# Simulate Network B
print("\nSimulating Network B...")
dpls_B = simulate_dipole(net_B, tstop=225.0, n_trials=1)

# Plot results
plt.figure(figsize=(12, 8))

# Plot Network A dipole
plt.subplot(2, 1, 1)
times_A = dpls_A[0].times
data_A = dpls_A[0].data['agg']
plt.plot(times_A, data_A, 'b-', label='Network A')
plt.title('Network A Dipole')
plt.xlabel('Time (ms)')
plt.ylabel('Dipole (nAm)')
plt.grid(True)
plt.legend()

# Plot Network B dipole
plt.subplot(2, 1, 2)
times_B = dpls_B[0].times
data_B = dpls_B[0].data['agg']
plt.plot(times_B, data_B, 'r-', label='Network B')
plt.title('Network B Dipole (receiving spikes from Network A)')
plt.xlabel('Time (ms)')
plt.ylabel('Dipole (nAm)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('network_communication.png')
print("Saved plot to 'network_communication.png'")
plt.show()

print("Complete!")