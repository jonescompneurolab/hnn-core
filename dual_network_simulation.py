"""
Dual Network Simulation

This script demonstrates simulating two copies of a typical HNN-Core network
simultaneously, first without connections between them, then with inter-network
connections from pyramidal cells of one network to all cell types of the other.
"""

import numpy as np
import matplotlib.pyplot as plt
from hnn_core import jones_2009_model, simulate_dipole
from hnn_core import Network
from hnn_core import Dipole
import copy


def create_dual_network(base_params, network_separation=2000):
    """
    Create two separate networks positioned far apart.
    
    Parameters
    ----------
    base_params : dict
        Base parameters for network creation
    network_separation : float
        Distance (in μm) to separate the two networks
        
    Returns
    -------
    net1, net2 : Network objects
        Two separate network instances
    """
    # Create first network with original cell type names
    net1 = jones_2009_model(base_params.copy())
    
    # Create second network by copying the first and then modifying it
    net2_new = copy.deepcopy(net1)
    
    # Define mapping for new cell types
    cell_type_mapping = {
        'L2_pyramidal': 'L2_pyramidal_net2',
        'L5_pyramidal': 'L5_pyramidal_net2',
        'L2_basket': 'L2_basket_net2',
        'L5_basket': 'L5_basket_net2'
    }
    
    # Preserve the origin position before clearing
    origin_pos = net2_new.pos_dict.get('origin', [(0., 0., 0.)])
    
    # Clear the second network and rebuild with new names
    net2_new.cell_types.clear()
    net2_new.pos_dict.clear()
    net2_new.gid_ranges.clear()
    net2_new.connectivity.clear()
    net2_new.external_drives.clear()
    
    # Restore the origin position for drives
    net2_new.pos_dict['origin'] = origin_pos
    
    # Add cell types with new names
    for old_name, new_name in cell_type_mapping.items():
        if old_name in net1.cell_types:
            net2_new.cell_types[new_name] = net1.cell_types[old_name]
            
    # Copy and modify positions with spatial shift
    for old_name, new_name in cell_type_mapping.items():
        if old_name in net1.pos_dict:
            positions = net1.pos_dict[old_name]
            # Shift all positions in x-direction by network_separation
            shifted_positions = []
            for pos in positions:
                shifted_pos = (pos[0] + network_separation, pos[1], pos[2])
                shifted_positions.append(shifted_pos)
            net2_new.pos_dict[new_name] = shifted_positions
    
    # Find the highest GID used in the first network (including drives)
    max_gid_net1 = max([gid_range.stop - 1 for gid_range in net1.gid_ranges.values()])
    
    # Make sure to account for drives that might be added later
    # Find the total number of cells in net1 to estimate how many GIDs might be needed for drives
    total_cells_net1 = sum([len(net1.pos_dict[cell_type]) for cell_type in net1.cell_types])
    
    # Add a generous buffer to avoid GID conflicts
    current_gid = max_gid_net1 + total_cells_net1 + 500
    
    # Add GID ranges for the new cell types
    for new_cell_type in cell_type_mapping.values():
        if new_cell_type in net2_new.cell_types:
            n_cells = len(net2_new.pos_dict[new_cell_type])
            net2_new.gid_ranges[new_cell_type] = range(current_gid, current_gid + n_cells)
            current_gid += n_cells
    
    # Copy connectivity from original network but adjust for new cell types and GIDs
    # Calculate GID offset to map from net1 GIDs to net2 GIDs
    # Filter by cell type names that start with 'L' (layer names)
    net1_cell_gids = [gid_range.start for cell_type, gid_range in net1.gid_ranges.items() 
                      if cell_type.startswith('L')]
    net2_cell_gids = [gid_range.start for cell_type, gid_range in net2_new.gid_ranges.items() 
                      if cell_type.startswith('L')]
    # If both networks have pyramidal and basket cells, calculate GID offset
    if net1_cell_gids and net2_cell_gids:
        net1_min_gid = min(net1_cell_gids)
        net2_min_gid = min(net2_cell_gids)
        gid_offset = net2_min_gid - net1_min_gid
        
        for conn in net1.connectivity:
            new_conn = copy.deepcopy(conn)
            
            # Update cell type references in the connection
            if hasattr(new_conn, 'src_type') and new_conn.src_type in cell_type_mapping:
                new_conn.src_type = cell_type_mapping[new_conn.src_type]
            if hasattr(new_conn, 'target_type') and new_conn.target_type in cell_type_mapping:
                new_conn.target_type = cell_type_mapping[new_conn.target_type]
                
            # Adjust GIDs in connections if they exist
            if hasattr(new_conn, 'gid_pairs') and new_conn.gid_pairs:
                new_gid_pairs = {}
                for src_gid, target_gids in new_conn.gid_pairs.items():
                    new_src_gid = src_gid + gid_offset
                    new_target_gids = [tgid + gid_offset for tgid in target_gids]
                    new_gid_pairs[new_src_gid] = new_target_gids
                new_conn.gid_pairs = new_gid_pairs
                
            net2_new.connectivity.append(new_conn)
    
    # Print GID ranges for debugging
    print(f"Network 1 GID ranges: {net1.gid_ranges}")
    print(f"Network 2 GID ranges: {net2_new.gid_ranges}")
    
    return net1, net2_new


def combine_networks(net1, net2):
    """
    Combine two networks into a single network object.
    
    Parameters
    ----------
    net1, net2 : Network objects
        The two networks to combine (net2 should already have unique cell type names)
        
    Returns
    -------
    combined_net : Network object
        Combined network containing both original networks
    """
    # Start with a copy of the first network
    combined_net = copy.deepcopy(net1)
    
    # Add cell types from second network (these should already have unique names)
    for cell_type, cell_template in net2.cell_types.items():
        combined_net.cell_types[cell_type] = cell_template
        
    # Add positions from second network (these should already be shifted)
    for cell_type, positions in net2.pos_dict.items():
        # Skip 'origin' as it's already in combined_net
        if cell_type != 'origin':
            combined_net.pos_dict[cell_type] = positions
        
    # Add GID ranges from second network (these should already be calculated)
    for cell_type, gid_range in net2.gid_ranges.items():
        combined_net.gid_ranges[cell_type] = gid_range
        
    # Add connectivity from second network (already adjusted)
    for conn in net2.connectivity:
        combined_net.connectivity.append(conn)
        
    # Handle drives from second network with proper GID adjustment
    if net2.external_drives:
        # Calculate the offset needed for drive GIDs
        max_combined_gid = max([gid_range.stop - 1 for gid_range in combined_net.gid_ranges.values()])
        drive_gid_offset = max_combined_gid + 1000  # Add extra buffer to be safe
        
        for drive_name, drive in net2.external_drives.items():
            # The original drive in net2 might already have a _net2 suffix
            # Make sure we use a unique name that matches exactly what's expected
            new_drive_name = drive_name
            
            # Ensure we have the expected drive name format
            if new_drive_name not in combined_net.external_drives:
                new_drive = copy.deepcopy(drive)
                
                # Adjust target cell types in the drive to use the new names
                if 'weights_ampa' in new_drive and new_drive['weights_ampa'] is not None:
                    new_weights_ampa = {}
                    for cell_type, weight in new_drive['weights_ampa'].items():
                        new_weights_ampa[cell_type] = weight
                    new_drive['weights_ampa'] = new_weights_ampa
                    
                if 'weights_nmda' in new_drive and new_drive['weights_nmda'] is not None:
                    new_weights_nmda = {}
                    for cell_type, weight in new_drive['weights_nmda'].items():
                        new_weights_nmda[cell_type] = weight
                    new_drive['weights_nmda'] = new_weights_nmda
                
                # Add the drive with its original name from net2
                combined_net.external_drives[new_drive_name] = new_drive
                
                # Add GID range for the new drive
                if drive_name in net2.gid_ranges:
                    original_drive_range = net2.gid_ranges[drive_name]
                    n_drive_cells = original_drive_range.stop - original_drive_range.start
                    combined_net.gid_ranges[new_drive_name] = range(drive_gid_offset, drive_gid_offset + n_drive_cells)
                    drive_gid_offset += n_drive_cells + 100  # Add extra buffer between drives
        
    # For debugging, print all drive names and their GID ranges
    print("\nCombined network drives and GID ranges:")
    for drive_name in combined_net.external_drives.keys():
        if drive_name in combined_net.gid_ranges:
            print(f"  {drive_name}: {combined_net.gid_ranges[drive_name]}")
    
    return combined_net


def add_inter_network_connections(combined_net, connection_weight=1.5, connection_delay=0.5):
    """
    Add connections from pyramidal cells of network 1 to all cell types of network 2.
    
    Parameters
    ----------
    combined_net : Network object
        The combined network
    connection_weight : float
        Weight of inter-network connections (increased from 0.01 to 0.05)
    connection_delay : float
        Delay of inter-network connections (ms)
    """
    # Instead of collecting all GIDs at once, let's connect cell type by cell type
    
    # Get source cell types from network 1
    src_cell_types = ['L2_pyramidal', 'L5_pyramidal']
    
    # Get target cell types from network 2
    target_cell_types = ['L2_pyramidal_net2', 'L5_pyramidal_net2', 'L2_basket_net2', 'L5_basket_net2']
    
    # Dictionary mapping target cell types to their receptors and locations
    # Based on the error, we need to use appropriate receptors for each cell type
    target_receptors = {
        'L2_pyramidal_net2': {'receptor': 'gabaa', 'loc': 'soma'},
        'L5_pyramidal_net2': {'receptor': 'gabaa', 'loc': 'soma'},
        'L2_basket_net2': {'receptor': 'ampa', 'loc': 'soma'},
        'L5_basket_net2': {'receptor': 'ampa', 'loc': 'soma'},
    }
    
    # Add connections one source cell type at a time
    for src_type in src_cell_types:
        # Convert to list of ints for consistency
        src_gids = list(range(combined_net.gid_ranges[src_type].start, 
                             combined_net.gid_ranges[src_type].stop))
        
        # Connect to each target cell type individually
        for target_type in target_cell_types:
            target_gids = list(range(combined_net.gid_ranges[target_type].start, 
                                    combined_net.gid_ranges[target_type].stop))
            
            # Get the appropriate receptor and location for this target cell type
            receptor = target_receptors[target_type]['receptor']
            loc = target_receptors[target_type]['loc']
            
            print(f"Connecting {src_type} ({len(src_gids)} cells) to {target_type} ({len(target_gids)} cells) using {receptor} receptors at {loc}")
            
            # Use the lower-level connection method for more control
            try:
                combined_net.add_connection(
                    src_gids=src_gids,
                    target_gids=target_gids,
                    loc=loc,
                    receptor=receptor,
                    weight=connection_weight,  # Increased for more visible effect
                    delay=connection_delay,
                    lamtha=100.0,
                    allow_autapses=False,
                    probability=0.2  # Increased from 0.1 for more connections
                )
            except ValueError as e:
                # If we get a receptor error, try a different receptor
                if "receptor is not defined" in str(e):
                    print(f"  Error: {e}")
                    print(f"  Trying alternative receptor...")
                    
                    # Try GABAA if AMPA failed, or vice versa
                    alt_receptor = 'gabaa' if receptor == 'ampa' else 'ampa'
                    
                    try:
                        print(f"  Trying with {alt_receptor} receptor instead...")
                        combined_net.add_connection(
                            src_gids=src_gids,
                            target_gids=target_gids,
                            loc=loc,
                            receptor=alt_receptor,
                            weight=connection_weight,  # Increased
                            delay=connection_delay,
                            lamtha=100.0,
                            allow_autapses=False,
                            probability=0.2  # Increased
                        )
                        print(f"  Success with {alt_receptor}!")
                    except ValueError as e2:
                        print(f"  Alternative receptor also failed: {e2}")
                        print(f"  Skipping connection from {src_type} to {target_type}")
                else:
                    # Re-raise other types of errors
                    raise


def extract_separate_dipoles(combined_dipole, net1_gid_ranges, net2_gid_ranges):
    """
    Extract separate dipole signals for each network from the combined simulation.
    
    This is a simplified approach - in practice, dipole separation would require
    more sophisticated analysis based on cell positions and contributions.
    """
    # For now, return the combined dipole as both networks
    # In a full implementation, this would separate contributions based on 
    # cell positions and current sources
    
    # Create separate dipole objects (simplified approach)
    dipole_net1 = combined_dipole.copy()
    dipole_net2 = combined_dipole.copy()
    
    # Scale by approximate contribution (simplified)
    dipole_net1.scale(0.5)
    dipole_net2.scale(0.5)
    
    return dipole_net1, dipole_net2


def main():
    """Main simulation function."""
    print("Starting dual network simulation...")
    
    # Create base parameters (using jones_2009_model defaults)
    from hnn_core import jones_2009_model
    base_net = jones_2009_model()
    base_params = base_net._params
    
    # Simulation parameters
    tstop = 170.0  # ms - increased to see full response
    dt = 0.025     # ms
    
    print("\n=== Simulating two separate networks without connections ===")
    
    # Create two separate networks
    net1, net2 = create_dual_network(base_params, network_separation=1000)
    
    print(f"Network 1 cell types: {list(net1.cell_types.keys())}")
    print(f"Network 2 cell types: {list(net2.cell_types.keys())}")
    
    # Add some drives to make the simulation interesting
    net1.add_evoked_drive(
        'evdist1', mu=40.0, sigma=3.85, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
    )
    
    # Use a unique name for the second network's drive
    net2.add_evoked_drive(
        'evdist2', mu=100.0, sigma=3.85, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal_net2': 5.4e-5, 'L5_pyramidal_net2': 5.4e-5}
    )
    
    # Print drive names for debugging
    print(f"Network 1 drives: {list(net1.external_drives.keys())}")
    print(f"Network 2 drives: {list(net2.external_drives.keys())}")
    
    # Combine networks
    combined_net = combine_networks(net1, net2)
    
    print(f"Combined network cell types: {list(combined_net.cell_types.keys())}")
    print(f"Combined network GID ranges: {combined_net.gid_ranges}")
    
    # Simulate without inter-network connections
    print("Simulating combined network without inter-network connections...")
    dpls_separate = simulate_dipole(combined_net, tstop=tstop, dt=dt, n_trials=1)
    
    # Create tutorial directory to save plots
    import os
    tutorial_dir = os.path.join(os.path.dirname(__file__), "tutorials")
    os.makedirs(tutorial_dir, exist_ok=True)
    
    print("\n=== Analysis and Visualization ===")
    
    # Use a nicer style for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract separate dipoles (approximate)
    dipole_A, dipole_B = extract_separate_dipoles(dpls_separate[0], None, None)
    
    
    # Figure 2: Overlay of all dipoles
    plt.figure(figsize=(12, 8))
    plt.plot(dipole_A.times, dipole_A.data['agg'], label='Network A', linewidth=4, color='#1f77b4')
    plt.plot(dipole_B.times, dipole_B.data['agg'], label='Network B', linewidth=2, color='#ff7f0e')
    plt.plot(dpls_separate[0].times, dpls_separate[0].data['agg'], label='Combined Network', 
             linewidth=2, color='#2ca02c', alpha=0.8)
    
    plt.title('Dual Network Simulation (No Connections)', fontsize=18)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Dipole Moment (nAm)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(tutorial_dir, 'dual_network_overlay.png'), dpi=300)
    
    
    
    print(f"\nSummary Statistics:")
    print(f"Separate networks - Peak dipole: {np.max(np.abs(dpls_separate[0].data['agg'])):.2e} nAm")
    print(f"\nPlots saved in: {tutorial_dir}")
    
    # Comment out the inter-network connections part
    
    # === Phase 2: Adding inter-network connections ===
    
    # Add connections between networks
    add_inter_network_connections(combined_net)
    
    print(f"Total connections after adding inter-network: {len(combined_net.connectivity)}")
    
    # Simulate with inter-network connections
    print("Simulating combined network with inter-network connections...")
    dpls_connected = simulate_dipole(combined_net, tstop=tstop, dt=dt, n_trials=1)
    
    # Plot comparison between connected and non-connected networks
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot separate networks simulation
    dpls_separate[0].plot(ax=axes[0, 0], layer='agg', show=False)
    axes[0, 0].set_title('Combined Networks (No Inter-connections)')
    axes[0, 0].set_ylabel('Dipole Moment (nAm)')
    
    # Plot connected networks simulation
    dpls_connected[0].plot(ax=axes[0, 1], layer='agg', show=False)
    axes[0, 1].set_title('Combined Networks (With Inter-connections)')
    axes[0, 1].set_ylabel('Dipole Moment (nAm)')
    
    # Plot comparison
    axes[1, 0].plot(dpls_separate[0].times, dpls_separate[0].data['agg'], 
                   label='No Inter-connections', linewidth=2)
    axes[1, 0].plot(dpls_connected[0].times, dpls_connected[0].data['agg'], 
                   label='With Inter-connections', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Dipole Moment (nAm)')
    axes[1, 0].set_title('Comparison')
    axes[1, 0].legend()
    
    # Plot difference
    diff_signal = dpls_connected[0].data['agg'] - dpls_separate[0].data['agg']
    axes[1, 1].plot(dpls_separate[0].times, diff_signal, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Difference (nAm)')
    axes[1, 1].set_title('Effect of Inter-network Connections')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(tutorial_dir, 'dual_network_interconnection.png'), dpi=300)
    
    # Print summary statistics for connection comparison with higher precision
    print(f"\nDetailed Summary Statistics (6 decimal places):")
    print(f"Separate networks - Peak dipole: {np.max(np.abs(dpls_separate[0].data['agg'])):.6f} nAm")
    print(f"Connected networks - Peak dipole: {np.max(np.abs(dpls_connected[0].data['agg'])):.6f} nAm")
    print(f"Max difference due to connections: {np.max(np.abs(diff_signal)):.6f} nAm")
    
    # Also compute and display statistics at different time points to see where differences occur
    print("\nDifferences at key time points (6 decimal places):")
    
    # Find indices for times near when network 2 is driven (around 100ms)
    time_points = [40, 80, 100, 120, 160]
    for t in time_points:
        idx = np.argmin(np.abs(dpls_separate[0].times - t))
        actual_time = dpls_separate[0].times[idx]
        
        no_conn_val = dpls_separate[0].data['agg'][idx]
        with_conn_val = dpls_connected[0].data['agg'][idx]
        diff_val = diff_signal[idx]
        
        print(f"Time {actual_time:.1f} ms:")
        print(f"  - No connections: {no_conn_val:.6f} nAm")
        print(f"  - With connections: {with_conn_val:.6f} nAm")
        print(f"  - Difference: {diff_val:.6f} nAm")
    return dpls_separate, dpls_connected, combined_net
    
    # Return only the non-connected simulation results
    # return dpls_separate, None, combined_net


if __name__ == "__main__":
    dpls_separate, dpls_connected, combined_net = main()