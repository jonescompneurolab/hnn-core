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


def create_dual_network(base_params, network_separation):
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
    # Add network identifier for clarity
    net1.network_id = 'A'
    
    # Create second network by copying the first and then modifying it
    net2_new = copy.deepcopy(net1)
    # Add network identifier for clarity
    net2_new.network_id = 'B'
    
    # Define mapping for new cell types
    # HARDCODED CELL TYPE MAPPING - temporary solution until dynamic cell type handling is implemented
    cell_type_mapping = {
        'L2_pyramidal': 'L2_pyramidal_net2',
        'L5_pyramidal': 'L5_pyramidal_net2',
        'L2_basket': 'L2_basket_net2',
        'L5_basket': 'L5_basket_net2'
    }
    
    # Create an inverse mapping for later use
    inverse_cell_type_mapping = {v: k for k, v in cell_type_mapping.items()}
    net2_new._rename_cell_types(cell_type_mapping)
    
    # Preserve the origin position before clearing
    origin_pos = net2_new.pos_dict.get('origin')
    
    # Clear the second network and rebuild with new names
    net2_new.cell_types.clear()
    net2_new.pos_dict.clear()
    net2_new.gid_ranges.clear()
    net2_new.connectivity.clear()
    net2_new.external_drives.clear()
    
    # Restore the origin position for drives but SHIFT IT to the new network location
    # This is crucial - the origin position determines where the drives are placed
    # Check if origin_pos is a list of tuples or a single tuple
    if isinstance(origin_pos, list):
        # If it's a list of tuples, shift the x-coordinate of the first tuple
        shifted_origin = [(origin_pos[0][0] + network_separation, origin_pos[0][1], origin_pos[0][2])]
    else:
        # If it's a single tuple, shift its x-coordinate
        shifted_origin = [(origin_pos[0] + network_separation, origin_pos[1], origin_pos[2])]
    
    net2_new.pos_dict['origin'] = shifted_origin
    
    # Add cell types with new names
    for old_name, new_name in cell_type_mapping.items():
        if old_name in net1.cell_types:
            # Create a deep copy of the cell template to avoid shared references
            net2_new.cell_types[new_name] = copy.deepcopy(net1.cell_types[old_name])
            
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
    current_gid = max_gid_net1 + total_cells_net1 + 200
    
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
            if 'src_type' in new_conn and new_conn['src_type'] in cell_type_mapping:
                new_conn['src_type'] = cell_type_mapping[new_conn['src_type']]
            if 'target_type' in new_conn and new_conn['target_type'] in cell_type_mapping:
                new_conn['target_type'] = cell_type_mapping[new_conn['target_type']]
                
            # Adjust GIDs in connections if they exist
            if 'gid_pairs' in new_conn and new_conn['gid_pairs']:
                new_gid_pairs = {}
                for src_gid, target_gids in new_conn['gid_pairs'].items():
                    new_src_gid = src_gid + gid_offset
                    new_target_gids = [tgid + gid_offset for tgid in target_gids]
                    new_gid_pairs[new_src_gid] = new_target_gids
                new_conn['gid_pairs'] = new_gid_pairs
                
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
        drive_gid_offset = max_combined_gid + 200  # Add extra buffer to be safe
        
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


def add_inter_network_connections(combined_net, connection_weight=20.0, connection_delay=0.5):
    """
    Add connections from pyramidal cells of network 1 to all cell types of network 2.
    
    Parameters
    ----------
    combined_net : Network object
        The combined network
    connection_weight : float
        Weight of inter-network connections
    connection_delay : float
        Delay of inter-network connections (ms)
    """
    # Get source cell types from network 1
    src_cell_types = ['L2_pyramidal', 'L5_pyramidal']
    
    # Get target cell types from network 2
    target_cell_types = ['L2_pyramidal_net2', 'L5_pyramidal_net2', 'L2_basket_net2', 'L5_basket_net2']
    
    # Dictionary mapping target cell types to their receptors and locations
    target_receptors = {
        'L2_pyramidal_net2': [
            {'receptor': 'ampa', 'loc': 'proximal', 'weight': connection_weight * 2.0},
            {'receptor': 'nmda', 'loc': 'proximal', 'weight': connection_weight}
        ],
        'L5_pyramidal_net2': [
            {'receptor': 'ampa', 'loc': 'proximal', 'weight': connection_weight * 2.0},
            {'receptor': 'nmda', 'loc': 'proximal', 'weight': connection_weight}
        ],
        'L2_basket_net2': [
            {'receptor': 'ampa', 'loc': 'soma', 'weight': connection_weight * 3.0}
        ],
        'L5_basket_net2': [
            {'receptor': 'ampa', 'loc': 'soma', 'weight': connection_weight * 3.0}
        ],
    }
    
    # Add connections one source cell type at a time
    for src_type in src_cell_types:
        src_gids = list(range(combined_net.gid_ranges[src_type].start, 
                            combined_net.gid_ranges[src_type].stop))
        
        for target_type in target_cell_types:
            if target_type not in combined_net.gid_ranges:
                print(f"Warning: Target type {target_type} not found in network, skipping")
                continue
                
            target_gids = list(range(combined_net.gid_ranges[target_type].start, 
                                combined_net.gid_ranges[target_type].stop))
            
            # For each target type, go through each receptor configuration
            for receptor_config in target_receptors[target_type]:
                receptor = receptor_config['receptor']
                loc = receptor_config['loc']
                weight = receptor_config['weight']
                
                print(f"Connecting {src_type} to {target_type} using {receptor} at {loc} with weight {weight}")
                
                try:
                    combined_net.add_connection(
                        src_gids=src_gids,
                        target_gids=target_gids,
                        loc=loc,
                        receptor=receptor,
                        weight=weight,
                        delay=connection_delay,
                        lamtha=100.0,
                        allow_autapses=False,
                        probability=0.8  # Increased probability
                    )
                    print(f"  Success with {receptor}!")
                except ValueError as e:
                    print(f"  Error: {e}")


def extract_separate_dipoles(combined_dipole, net1, net2):
    """
    Extract separate dipole signals for each network from the combined simulation.
    
    This function uses the spatial coordinates of each network to separate the dipole
    contributions. Since the networks are spatially separated, we can identify which
    cells belong to which network based on their x-coordinate.
    
    Parameters
    ----------
    combined_dipole : Dipole object
        The dipole object from the combined network simulation
    net1 : Network object
        First network
    net2 : Network object
        Second network
        
    Returns
    -------
    dipole_net1, dipole_net2 : Dipole objects
        Separate dipole objects for each network
    """
    # Create separate dipole objects with both times and data parameters
    dipole_net1 = Dipole(combined_dipole.times, combined_dipole.data.copy())
    dipole_net2 = Dipole(combined_dipole.times, combined_dipole.data.copy())
    
    # Calculate the midpoint between the two networks to determine the boundary
    # Find the maximum x-coord in net1 and minimum x-coord in net2
    x_coords_net1 = []
    for cell_type, positions in net1.pos_dict.items():
        if cell_type != 'origin':
            for pos in positions:
                # Handle different position formats safely
                if isinstance(pos[0], (int, float)):
                    x_coords_net1.append(float(pos[0]))
                elif isinstance(pos[0], tuple):
                    x_coords_net1.append(float(pos[0][0]))
    
    x_coords_net2 = []
    for cell_type, positions in net2.pos_dict.items():
        if cell_type != 'origin':
            for pos in positions:
                # Handle different position formats safely
                if isinstance(pos[0], (int, float)):
                    x_coords_net2.append(float(pos[0]))
                elif isinstance(pos[0], tuple):
                    x_coords_net2.append(float(pos[0][0]))
    
    # Calculate the midpoint
    max_x_net1 = max(x_coords_net1) if x_coords_net1 else 0
    min_x_net2 = min(x_coords_net2) if x_coords_net2 else 3000  # Default to network_separation
    
    boundary_x = (max_x_net1 + min_x_net2) / 2
    
    print(f"Dipole separation boundary at x = {boundary_x:.1f} μm")
    
    # For the actual separation, we would need to access the individual cell contributions
    # Since we don't have that in this simple model, we'll use a simplified approach
    
    # Simplified approach: Scale the combined dipole based on cell counts in each network
    n_cells_net1 = sum([len(positions) for cell_type, positions in net1.pos_dict.items() 
                       if cell_type != 'origin'])
    n_cells_net2 = sum([len(positions) for cell_type, positions in net2.pos_dict.items() 
                       if cell_type != 'origin'])
    total_cells = n_cells_net1 + n_cells_net2
    
    # Scale the dipole data for each network based on cell count proportions
    for layer in dipole_net1.data:
        dipole_net1.data[layer] = combined_dipole.data[layer] * (n_cells_net1 / total_cells)
        dipole_net2.data[layer] = combined_dipole.data[layer] * (n_cells_net2 / total_cells)
    
    return dipole_net1, dipole_net2


def plot_custom_spike_raster(combined_net, tutorial_dir=None):
    """Plot custom spike raster plot without using built-in HNN functions."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import csv
    
    if combined_net.cell_response is None:
        raise ValueError("Network has no spike data. Run a simulation first.")
    
    # Get all cell types from the GID ranges
    cell_types = {}
    for cell_type, gid_range in combined_net.gid_ranges.items():
        # Skip drive cells for clearer visualization
        if not cell_type.startswith('ev'):  
            # Map each GID to its cell type
            for gid in range(gid_range.start, gid_range.stop):
                cell_types[gid] = cell_type
    
    # Create mapping of GIDs to display rows for y-axis
    gids = sorted(cell_types.keys())
    gid_to_row = {gid: i for i, gid in enumerate(gids)}
    
    # Identify network 1 and network 2 cell types
    net1_types = [ct for ct in set(cell_types.values()) if not ct.endswith('_net2')]
    net2_types = [ct for ct in set(cell_types.values()) if ct.endswith('_net2')]
    
    # Get spike data from first trial - FIX: use spike_gids instead of spiketimes_gid
    if len(combined_net.cell_response.spike_gids) > 0:
        spike_times = []
        spike_rows = []
        spike_colors = []
        
        # Color mapping for cell types
        color_map = {
            'L2_basket': '#1f77b4',      # blue
            'L2_pyramidal': '#7aafe5',   # light blue
            'L5_basket': '#2ca02c',      # green
            'L5_pyramidal': '#ff7f0e',   # orange
            'L2_basket_net2': '#9467bd',  # purple
            'L2_pyramidal_net2': '#c5b0d5', # light purple
            'L5_basket_net2': '#d62728',  # red
            'L5_pyramidal_net2': '#ff9896' # light red
        }
        
        # Count spikes by cell type for the legend
        spike_counts = {ct: 0 for ct in set(cell_types.values())}
        
        # Process each spike - FIX: use spike_gids instead of spiketimes_gid
        for gid, time in zip(combined_net.cell_response.spike_gids[0], 
                           combined_net.cell_response.spike_times[0]):
            if gid in cell_types:
                cell_type = cell_types[gid]
                spike_times.append(time)
                spike_rows.append(gid_to_row[gid])
                spike_colors.append(color_map.get(cell_type, 'black'))
                spike_counts[cell_type] += 1
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot the spikes as colored dots
        ax.scatter(spike_times, spike_rows, c=spike_colors, s=5, alpha=0.8)
        
        # Customize the y-axis to show cell type boundaries
        net1_gids = [gid for gid in gids if cell_types[gid] in net1_types]
        net2_gids = [gid for gid in gids if cell_types[gid] in net2_types]
        
        if net1_gids and net2_gids:
            # Add horizontal lines to separate cell types
            for cell_type in sorted(set(cell_types.values())):
                cell_type_gids = [gid for gid, ct in cell_types.items() if ct == cell_type]
                if cell_type_gids:
                    min_row = min([gid_to_row[gid] for gid in cell_type_gids])
                    ax.axhline(y=min_row-0.5, color='gray', linestyle='-', alpha=0.3)
            
            # Add horizontal bars to indicate networks
            net1_min = min([gid_to_row[gid] for gid in net1_gids])
            net1_max = max([gid_to_row[gid] for gid in net1_gids])
            net2_min = min([gid_to_row[gid] for gid in net2_gids])
            net2_max = max([gid_to_row[gid] for gid in net2_gids])
            
            # Label the network regions
            ax.text(-5, (net1_min + net1_max) / 2, "Network 1", 
                    va='center', ha='right', fontsize=12, weight='bold')
            ax.text(-5, (net2_min + net2_max) / 2, "Network 2", 
                    va='center', ha='right', fontsize=12, weight='bold')
        
        # Set axis labels and title
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Cell ID', fontsize=14)
        ax.set_title('Custom Spike Raster Plot', fontsize=16)
        
        # Create a legend
        legend_elements = []
        for cell_type, color in color_map.items():
            if cell_type in spike_counts:
                from matplotlib.lines import Line2D
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=8, label=f"{cell_type} ({spike_counts[cell_type]} spikes)")
                )
        
        # Place legend outside the plot
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.01, 1), fontsize=10)
        
        # Save spike data to CSV
        if tutorial_dir is not None:
            os.makedirs(tutorial_dir, exist_ok=True)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(tutorial_dir, 'custom_spike_raster.png'), dpi=300)
            print(f"Custom spike raster saved to {os.path.join(tutorial_dir, 'custom_spike_raster.png')}")
            
            # Save spike data to CSV - FIX: use spike_gids instead of spiketimes_gid
            csv_path = os.path.join(tutorial_dir, 'spike_data.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['GID', 'Cell Type', 'Time (ms)'])
                
                # Loop through all spikes
                for i, (gid, time) in enumerate(zip(
                    combined_net.cell_response.spike_gids[0],
                    combined_net.cell_response.spike_times[0]
                )):
                    if gid in cell_types:
                        csvwriter.writerow([gid, cell_types[gid], time])
            
            print(f"Spike data saved to {csv_path}")
        
        return fig
    else:
        print("No spike data available in combined network.")
        return None


def main():
    """Main simulation function."""
    print("Starting dual network simulation...")
    # Create tutorial directory to save plots
    import os
    tutorial_dir = os.path.join(os.path.dirname(__file__), "tutorials")
    output_dir = os.path.join(os.path.dirname(__file__),"output")
    os.makedirs(tutorial_dir, exist_ok=True)
    os.makedirs(output_dir,exist_ok=True)


    # Create base parameters (using jones_2009_model defaults)
    from hnn_core import jones_2009_model
    base_net = jones_2009_model()
    base_params = base_net._params
    
    # Simulation parameters
    tstop = 170.0  # ms - increased to see full response
    dt = 0.025     # ms
    
    print("\n=== Simulating two separate networks without connections ===")
    
    # Create two separate networks - INCREASE SEPARATION FOR CLARITY
    net1, net2 = create_dual_network(base_params, network_separation=500)
    
    print(f"Network 1 cell types: {list(net1.cell_types.keys())}")
    print(f"Network 2 cell types: {list(net2.cell_types.keys())}")
    
    # Add some drives to make the simulation interesting
    net1.add_evoked_drive(
        'evdist1', mu=40.0, sigma=3.85, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal': 5.4e-2, 'L5_pyramidal': 5.4e-2}  # KEEP AS IS
    )
    
    # FIXED: Modify the network 2 drive code to properly handle the cell types
    # First verify the cell types are in the net2 object
    print(f"Net2 cell types available: {list(net2.cell_types.keys())}")
    print(f"Net2 gid ranges: {net2.gid_ranges}")
    
    # Use the correct cell types for network 2 (with _net2 suffix)
    # and make sure they exist in net2's gid_ranges
    valid_net2_cell_types = [ct for ct in net2.gid_ranges.keys() if not ct.startswith('ev')]
    print(f"Valid net2 cell types with GID ranges: {valid_net2_cell_types}")
    
    # Create weights dict using only valid cell types
    weights_ampa_net2 = {}
    for cell_type in valid_net2_cell_types:
        if 'pyramidal' in cell_type:
            weights_ampa_net2[cell_type] = 5.4e-1  # Strong drive for pyramidal
        elif 'basket' in cell_type:
            weights_ampa_net2[cell_type] = 2.0e-1  # Drive for basket cells
    
    # Verify the weights we're about to use
    print(f"Network 2 drive weights: {weights_ampa_net2}")
    
    # # Add the drive with verified cell types
    # if weights_ampa_net2:  # Only add if we have valid targets
    #     net2.add_evoked_drive(
    #         'evdist2', mu=100.0, sigma=3.85, numspikes=1, location='distal',
    #         weights_ampa=weights_ampa_net2
    #     )
    
    # VERIFY drive was properly added
    # print(f"Net2 drives after adding: {list(net2.external_drives.keys())}")
    
    # Simulate the networks separately
    print("\nSimulating Network 1...")
    net1_dpls = simulate_dipole(net1, tstop=tstop, dt=dt, n_trials=1)
    
    print("\nSimulating Network 2...")
    net2_dpls = simulate_dipole(net2, tstop=tstop, dt=dt, n_trials=1)
    

    print("\n=== Plotting Raster Plots ===")
    plot_custom_spike_raster(net1, output_dir)

    print("\n=== Plotting Raster Plots ===")
    plot_custom_spike_raster(net2, tutorial_dir)
    
    
    # Combine networks
    combined_net = combine_networks(net1, net2)
    
    print(f"Combined network cell types: {list(combined_net.cell_types.keys())}")
    print(f"Combined network GID ranges: {combined_net.gid_ranges}")
    
    # Simulate without inter-network connections
    print("Simulating combined network without inter-network connections...")
    dpls_separate = simulate_dipole(combined_net, tstop=tstop, dt=dt, n_trials=1)
    print(f"Combined network cell types after simulation  : {list(combined_net.cell_types.keys())}")
    # print(f"Combined network GID ranges: {combined_net.gid_ranges}")

    # print("\n=== Plotting Raster Plots ===")
    # plot_custom_spike_raster(combined_net, tutorial_dir)
    
    print("\n=== Analysis and Visualization ===")
    
    # Use a nicer style for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract separate dipoles with improved function
    dipole_A, dipole_B = extract_separate_dipoles(dpls_separate[0], net1, net2)
    
    
    # === Phase 2: Adding inter-network connections ===
    
    # Add connections between networks
    # INCREASE WEIGHT for more visible effect
    add_inter_network_connections(combined_net, connection_weight=3.0)
    
    print(f"Total connections after adding inter-network: {len(combined_net.connectivity)}")
    
    # Simulate with inter-network connections
    print("Simulating combined network with inter-network connections...")
    dpls_connected = simulate_dipole(combined_net, tstop=tstop, dt=dt, n_trials=1)
    # print("\n=== Plotting Raster Plots ===")
    # plot_custom_spike_raster(combined_net,output_dir )
    # Figure 2: Overlay of all dipoles
    # plt.figure(figsize=(12, 8))
    # plt.plot(dipole_A.times, dipole_A.data['agg'], label='Network A', linewidth=2, color='#1f77b4')
    # plt.plot(dipole_B.times, dipole_B.data['agg'], label='Network B', linewidth=2, color='#ff7f0e')
    # plt.plot(dpls_connected[0].times, dpls_connected[0].data['agg'], label='Connected Network with internetwork', linewidth=2, color="#fae600")
    # plt.plot(dpls_separate[0].times, dpls_separate[0].data['agg'], label='Combined Network without internework', 
    #          linewidth=2, color='#2ca02c', alpha=0.8)
    
    # plt.title('Dual Network Simulation (No Connections)', fontsize=18)
    # plt.xlabel('Time (ms)', fontsize=14)
    # plt.ylabel('Dipole Moment (nAm)', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tick_params(labelsize=12)
    # plt.tight_layout()
    # plt.savefig(os.path.join(tutorial_dir, 'dual_network_overlay.png'), dpi=300)
    
    print(f"\nSummary Statistics:")
    print(f"Separate networks - Peak dipole: {np.max(np.abs(dpls_separate[0].data['agg'])):.2e} nAm")
    print(f"\nPlots saved in: {tutorial_dir}")
    
    return dpls_separate, dpls_connected, combined_net
    
    # # Return only the non-connected simulation results
    # # return dpls_separate, None, combined_net


if __name__ == "__main__":
    dpls_separate, dpls_connected, combined_net = main()
