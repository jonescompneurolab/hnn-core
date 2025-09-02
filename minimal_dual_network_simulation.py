"""
Minimal Dual Network Simulation

Simulates two smallest possible HNN-Core networks (one cell per type) and analyzes results.
"""
import copy
from collections import OrderedDict
from hnn_core import jones_2009_model, simulate_dipole, Network
import matplotlib.pyplot as plt
from hnn_core.hnn_io import write_network_configuration
from hnn_core.viz import plot_dipole
from hnn_core.viz import plot_dipole

def create_minimal_network(cell_type_suffix=None, gid_start=0, network_separation=0):
    """Create a minimal network with one cell per type and shift positions."""
    print("Network creation start")
    net = jones_2009_model()
    net.suffix = cell_type_suffix
    # Optionally rename cell types for net2
    if cell_type_suffix:
        original_gid_ranges = copy.deepcopy(net.gid_ranges)
        print("Suffix in main ",net.suffix)
        mapping = OrderedDict([
            ('L2_basket', f'L2_basket{cell_type_suffix}'),
            ('L2_pyramidal', f'L2_pyramidal{cell_type_suffix}'),
            ('L5_basket', f'L5_basket{cell_type_suffix}'),
            ('L5_pyramidal', f'L5_pyramidal{cell_type_suffix}'),
        ])
        net._rename_cell_types(mapping)
        # # print("DEBUG: net pos dict: ", net.pos_dict)
        # # print("Debug: cell types: ",net.cell_types)
        # Update GID ranges to start at gid_start and avoid overlap
        current_gid = gid_start
        net.gid_ranges = OrderedDict()
        for ct in mapping.values():
            n_cells = len(net.pos_dict[ct])
            net.gid_ranges[ct] = range(current_gid, current_gid + n_cells)
            current_gid += n_cells
        # # print("Debug: net gid ranges: ",net.gid_ranges)
                # Update connectivity GIDs to match new gid_ranges
        for conn in net.connectivity:
            # Update target_gids if target_type is in mapping
            if 'target_type' in conn and conn['target_type'] in mapping.values():
                ct = conn['target_type']
                gid_range = list(net.gid_ranges[ct])
                n_targets = len(conn['target_gids'])
                # Replace with correct GIDs from gid_ranges
                conn['target_gids'] = gid_range[:n_targets]
            # Update src_gids if src_type is in mapping
            if 'src_type' in conn and conn['src_type'] in mapping.values():
                ct = conn['src_type']
                gid_range = list(net.gid_ranges[ct])
                n_srcs = len(conn['src_gids'])
                conn['src_gids'] = gid_range[:n_srcs]
            
            if 'gid_pairs' in conn and conn['gid_pairs']:
                src_type = conn['src_type']
                tgt_type = conn['target_type']
                # Get original cell types from mapping
                orig_src_type = [k for k, v in mapping.items() if v == src_type][0]
                orig_tgt_type = [k for k, v in mapping.items() if v == tgt_type][0]
                
                # Calculate offsets using original and new GID ranges
                src_offset = net.gid_ranges[src_type][0] - original_gid_ranges[orig_src_type][0]
                tgt_offset = net.gid_ranges[tgt_type][0] - original_gid_ranges[orig_tgt_type][0]

                # print(f"Debug: Updating gid_pairs for {src_type} -> {tgt_type}")
                # print(f"Debug: Original src range: {original_gid_ranges[orig_src_type]}")
                # print(f"Debug: New src range: {net.gid_ranges[src_type]}")
                # print(f"Debug: src_offset: {src_offset}, tgt_offset: {tgt_offset}")
                # print(f"Debug: Updating gid_pairs for {src_type} -> {tgt_type}")
                # print(f"Debug: Original src range: {original_gid_ranges[orig_src_type]}")
                # print(f"Debug: New src range: {net.gid_ranges[src_type]}")
                # print(f"Debug: src_offset: {src_offset}, tgt_offset: {tgt_offset}")
                
                new_gid_pairs = {}
                for src_gid, tgt_gids in conn['gid_pairs'].items():
                    new_src_gid = int(src_gid) + src_offset
                    new_tgt_gids = [int(tg) + tgt_offset for tg in tgt_gids]
                    new_gid_pairs[new_src_gid] = new_tgt_gids
                conn['gid_pairs'] = new_gid_pairs

            # print(f"Debug: Connection from {conn['src_type']} to {conn['target_type']}")
            # print(f"Debug: Receptor: {conn.get('receptor', 'not specified')}")
            # print(f"Debug: Location: {conn.get('loc', 'not specified')}")
            # print(f"Debug: Connection from {conn['src_type']} to {conn['target_type']}")
            # print(f"Debug: Receptor: {conn.get('receptor', 'not specified')}")
            # print(f"Debug: Location: {conn.get('loc', 'not specified')}")

    return net

def plot_combined_spike_raster(net1, net2, title="Combined Spike Raster",plot_net1=True, plot_net2=True):
    """Plot spikes from both networks on a single raster plot."""
    # Gather spike data from both networks
    spikes = []
    gids = []
    types = []
    # # Net1
    if plot_net1:
        for trial_idx in range(len(net1.cell_response.spike_times)):
            spikes.extend(net1.cell_response.spike_times[trial_idx])
            gids.extend(net1.cell_response.spike_gids[trial_idx])
            types.extend(net1.cell_response.spike_types[trial_idx])
    if plot_net2:
    # Net2
        for trial_idx in range(len(net2.cell_response.spike_times)):
            spikes.extend(net2.cell_response.spike_times[trial_idx])
            gids.extend(net2.cell_response.spike_gids[trial_idx])
            types.extend(net2.cell_response.spike_types[trial_idx])

    # Assign each cell type a color
    unique_types = sorted(set(types))
    color_map = {ct: plt.cm.tab10(i % 10) for i, ct in enumerate(unique_types)}
    colors = [color_map[ct] for ct in types]

    # Assign each GID a row for plotting
    unique_gids = sorted(set(gids))
    gid_to_row = {gid: i for i, gid in enumerate(unique_gids)}
    rows = [gid_to_row[gid] for gid in gids]

    plt.figure(figsize=(10, 6))
    plt.scatter(spikes, rows, c=colors, s=10)
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell (GID)")
    plt.title(title)
    # Add legend for cell types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=ct,
                   markerfacecolor=color_map[ct], markersize=8)
        for ct in unique_types
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()





def main():
    print("Minimal dual network simulation...")

    net1 = create_minimal_network(cell_type_suffix=None, gid_start=0)
    next_gid = net1.get_next_gid()
    net1.add_evoked_drive(
        'evdist1', mu=5.0, sigma=1.0, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1},gid_start=next_gid
    )
    next_gid = net1.get_next_gid()
    net2 = create_minimal_network(cell_type_suffix="_net2",gid_start=next_gid)    
    # Specify which cell types to use for dipole calculation
    net2.dipole_cell_types = ['L2_pyramidal_net2', 'L5_pyramidal_net2']
    # Only call get_next_gid(net2) here, before adding any drives!
    next_gid_net2 = net2.get_next_gid()
    net2.add_evoked_drive(
        'evdist2', mu=5.0, sigma=1.0, numspikes=1, location='distal',
        weights_ampa={'L2_pyramidal_net2': 0.1,'L5_pyramidal_net2': 0.1},gid_start=next_gid_net2
    )
    
    # After creating net1 and net2, or after combining:
    write_network_configuration(net1, "net1_config.json")
    write_network_configuration(net2, "net2_config.json")
    print("Simulating net1...")
    print("Net1 gid ranges:", net1.gid_ranges)
    dpl_1 = simulate_dipole(net1, tstop=20, dt=0.025, n_trials=1)
    print("debug dipole net1  type: ",type(dpl_1[0]))
    print("debug dipole net1  type: ",type(dpl_1[0]))
    print("Simulating net2...")
    print("Net2 gid ranges:", net2.gid_ranges)
    dpl_2 = simulate_dipole(net2, tstop=20, dt=0.025, n_trials=1)
    print("before connection")

    print("Net1 dipole peak:", max(abs(dpl_1[0].data['agg'])))
    print("Net2 dipole peak:", max(abs(dpl_2[0].data['agg'])))
    print("Net1 spikes:", len(net1.cell_response.spike_times[0]))
    print("Net2 spikes:", len(net2.cell_response.spike_times[0]))
    for ct in net1.cell_types:
        gids = list(net1.gid_ranges[ct])
        spikes = [gid for gid in gids if gid in net1.cell_response.spike_gids[0]]
        print(f"{ct}: {len(spikes)} spiking cells out of {len(gids)}")

    for ct in net2.cell_types:
        gids = list(net2.gid_ranges[ct])
        spikes = [gid for gid in gids if gid in net2.cell_response.spike_gids[0]]
        print(f"{ct}: {len(spikes)} spiking cells out of {len(gids)}")

    plot_combined_spike_raster(net1,net2, title="Net 1 + Net 2 ",plot_net1=True,plot_net2=True)

    # Plot dipoles next to each other
    dpls = [dpl_1[0], dpl_2[0]]
    from hnn_core.viz import plot_dipole
    plot_dipole(dpls, show=True)



if __name__ == "__main__":
    main()
