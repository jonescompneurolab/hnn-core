"""
==============================================
10. Simulating Two Minimal Independent Networks
==============================================

This how-to demonstrates constructing and running two *independent* minimal
HNN-Core networks side-by-side. Each network contains one cell per canonical
cell type from ``jones_2009_model``. The second network is the same, except it uses
disjoint GID ranges and remapped connectivity indices. This how-to is intended for
*future development* on simulating multiple `Network`s simultaneously, inside the same
simulation; hence, this uses non-public methods of `Network`.

Overview
--------
1. Build Network 1 (net1) from the common `jones_2009_model()`.
2. Add an Evoked Drive to net1.
3. Build Network 2 (net2) with non-overlapping GIDs.
4. Add an identical Evoked Drive to net2.
5. Simulate each network independently.
6. Compare dipole amplitudes and visualize spike rasters jointly.

Notes
-----
- This is a *prototype* scaffold: no cross-network synapses, and no unique celltype names.
- Only independent simulation is shown; simultaneous simulation or coupling between
  networks is not yet possible.

"""

# Authors:
# - Maira Usman <maira.usman5703@gmail.com>
# - Austin Soplata <me@asoplata.com>

###############################################################################
# Step 1: Imports
# ---------------

import matplotlib.pyplot as plt

from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.hnn_io import write_network_configuration
from hnn_core.viz import plot_dipole


###############################################################################
# Step 2: Helper to Build a Minimal Network
# -----------------------------------------
#
# Creates a network with one cell per canonical cell type. Their GIDs reassigned to
# avoid overlap with previously instantiated networks. Connectivity references
# (``src_gids``, ``target_gids``, and ``gid_pairs``) are updated accordingly.

def create_minimal_network(gid_start=0):
    """Create a minimal network with one cell per type and shift positions."""
    print("Network creation start")
    net = jones_2009_model()
    net._shift_gid_ranges(gid_start)
    return net


###############################################################################
# Step 3: Build First Network
# ---------------------------

net1 = create_minimal_network(gid_start=0)

###############################################################################
# Step 4: Add An Evoked Drive to net1
# -----------------------------------
# Each network receives a single distal evoked drive with matching timing
# parameters but distinct names.

drive_gid_1 = net1._get_next_available_gid()
net1.add_evoked_drive(
    'evdist1', mu=5.0, sigma=1.0, numspikes=1, location='distal',
    weights_ampa={'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1},
    gid_start=drive_gid_1
)

###############################################################################
# Step 5: Build Second Network Using Disjoint GIDs
# ------------------------------------------------
next_gid = net1._get_next_available_gid()
net2 = create_minimal_network(gid_start=next_gid)

###############################################################################
# Step 6: Add An Evoked Drive to net2
# -----------------------------------
drive_gid_2 = net2._get_next_available_gid()
net2.add_evoked_drive(
    'evdist2', mu=5.0, sigma=1.0, numspikes=1, location='distal',
    weights_ampa={'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1},
    gid_start=drive_gid_2
)

print("Net1 gid ranges:", net1.gid_ranges)
print("Net2 gid ranges:", net2.gid_ranges)

print("Next GIDs after drives:", net1._get_next_available_gid(), net2._get_next_available_gid())

###############################################################################
# Step 7: (Optional) Export Configurations
# ----------------------------------------

write_network_configuration(net1, "net1_config.json")
write_network_configuration(net2, "net2_config.json")
print("Configuration files written (net1_config.json, net2_config.json).")

###############################################################################
# Step 8: Simulate Networks Independently
# ---------------------------------------

dpl1 = simulate_dipole(net1, tstop=20.0, dt=0.025, n_trials=1)
dpl2 = simulate_dipole(net2, tstop=20.0, dt=0.025, n_trials=1)

print("Net1 dipole peak:", max(abs(dpl1[0].data['agg'])))
print("Net2 dipole peak:", max(abs(dpl2[0].data['agg'])))
print("Net1 spikes (trial 0):", len(net1.cell_response.spike_times[0]))
print("Net2 spikes (trial 0):", len(net2.cell_response.spike_times[0]))

###############################################################################
# Step 9: Plot Dipoles
# --------------------
dpls = [dpl1[0], dpl2[0]]
plot_dipole(dpls, show=True)

###############################################################################
# Step 10: Combined Spike Raster
# ------------------------------

def plot_combined_spike_raster(net_a, net_b, title="Combined Spike Raster"):
    """Aggregate spike rasters from both networks into a single figure."""
    spikes, gids, types = [], [], []
    for trial in range(len(net_a.cell_response.spike_times)):
        spikes.extend(net_a.cell_response.spike_times[trial])
        gids.extend(net_a.cell_response.spike_gids[trial])
        types.extend(net_a.cell_response.spike_types[trial])
    for trial in range(len(net_b.cell_response.spike_times)):
        spikes.extend(net_b.cell_response.spike_times[trial])
        gids.extend(net_b.cell_response.spike_gids[trial])
        types.extend(net_b.cell_response.spike_types[trial])

    if not spikes:
        print("No spikes to plot.")
        return

    unique_types = sorted(set(types))
    color_map = {ct: plt.cm.tab10(i % 10) for i, ct in enumerate(unique_types)}
    colors = [color_map[ct] for ct in types]

    unique_gids = sorted(set(gids))
    gid_to_row = {g: i for i, g in enumerate(unique_gids)}
    rows = [gid_to_row[g] for g in gids]

    plt.figure(figsize=(9, 5))
    plt.scatter(spikes, rows, c=colors, s=10)
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell (GID index)")
    plt.title(title)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=ct,
                   markerfacecolor=color_map[ct], markersize=6)
        for ct in unique_types
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


plot_combined_spike_raster(net1, net2, title="Net1 + Net2 Spikes")


