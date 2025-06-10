import hnn_core
from hnn_core import read_params, Network, simulate_dipole
import os.path as op
import csv

# Network A
params_A = read_params(op.join(hnn_core.__path__[0], 'param', 'default.json'))
params_A.update({'tstop': 50.0})
net_A = Network(params_A)
net_A.add_evoked_drive(
    'ev_A_1', mu=10.0, sigma=0.1, numspikes=1,
    weights_ampa={'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01},
    location='distal', synaptic_delays={'L2_pyramidal': 5.0, 'L5_pyramidal': 5.0}
)
dpls_A = simulate_dipole(net_A, tstop=50.0, n_trials=1)

# Extract and save spikes to CSV
with open('neuron_spikes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gid', 'cell_type', 'time_ms'])
    if net_A.cell_response and net_A.cell_response.spike_times:
        trial_idx = 0
        for i in range(len(net_A.cell_response.spike_times[trial_idx])):
            cell_type = net_A.cell_response.spike_types[trial_idx][i]
            if cell_type in ['L2_pyramidal', 'L5_pyramidal']:
                writer.writerow([
                    net_A.cell_response.spike_gids[trial_idx][i],
                    cell_type,
                    net_A.cell_response.spike_times[trial_idx][i]
                ])
print("Saved neuron spikes to neuron_spikes.csv")

# Format for Network B
spike_data = {}
target_config = {}
with open('neuron_spikes.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        src_id = f"NetA_{row['cell_type']}_GID{int(float(row['gid']))}"
        if src_id not in spike_data:
            spike_data[src_id] = []
        spike_data[src_id].append(float(row['time_ms']))
        target_config[src_id] = {
            'target_cell_types': ['L2_pyramidal'],
            'location': 'distal',
            'weights_ampa': {'L2_pyramidal': 0.005},
            'synaptic_delays': {'L2_pyramidal': 2.0},
            'probability': 1.0
        }

# Network B
params_B = read_params(op.join(hnn_core.__path__[0], 'param', 'default.json'))
params_B.update({'tstop': 100.0})
net_B = Network(params_B)

if spike_data:
    print(f"Number of source channels from A: {len(spike_data)}")
    net_B.add_spike_train_drive('drive_from_NetA', spike_data=spike_data, target_config=target_config)
    print("Added spike train drive to Network B. Verifying configuration...")

    # --- Verification ---
    drive_name = 'drive_from_NetA'
    if drive_name in net_B.external_drives:
        print(f"  Drive '{drive_name}' found in net_B.external_drives.")
        print(f"    Type: {net_B.external_drives[drive_name]['type']}")
        print(f"    Number of drive cells (source channels): {net_B.external_drives[drive_name]['n_drive_cells']}")
        # print(f"    Explicit spike data (first few entries): {list(net_B.external_drives[drive_name]['explicit_spike_data'].items())[:2]}") # Can be long

        if drive_name in net_B.gid_ranges:
            print(f"  GIDs for '{drive_name}' in net_B.gid_ranges: {net_B.gid_ranges[drive_name]}")
        else:
            print(f"  ERROR: GID range for '{drive_name}' NOT found in net_B.gid_ranges.")

        print(f"\n  Connectivity involving '{drive_name}' sources in net_B.connectivity:")
        drive_src_gids = set(net_B.gid_ranges.get(drive_name, []))
        found_connections = False
        for conn_idx, conn in enumerate(net_B.connectivity):
            if conn['src_gids'].intersection(drive_src_gids): # Check if any source GID from this connection is part of our drive
                found_connections = True
                print(f"    Connection {conn_idx}:")
                print(f"      Source GIDs from drive: {conn['src_gids'].intersection(drive_src_gids)}")
                print(f"      Target Type: {conn['target_type']}, Receptor: {conn['receptor']}, Loc: {conn['loc']}")
                print(f"      Weight: {conn['nc_dict']['A_weight']}, Delay: {conn['nc_dict']['A_delay']}")
        if not found_connections:
            print("    No connections found originating from the drive GIDs.")
    else:
        print(f"  ERROR: Drive '{drive_name}' not found in net_B.external_drives.")
    print("--- End of configuration verification ---")
else:
    print("No spike data to add as drive to Network B.")

# Add some intrinsic drive to Network B to see if it spikes on its own or with A's input
net_B.add_evoked_drive(
    'ev_B_local', mu=30, sigma=1, numspikes=1, location='proximal',
    weights_ampa={'L5_pyramidal': 0.01}, synaptic_delays=0.1
)

print("\nSimulating Network B...")
dpls_B = simulate_dipole(net_B, tstop=100.0, n_trials=1)
print("Network B simulation complete")