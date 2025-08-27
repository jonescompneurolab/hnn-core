import numpy as np
import sys
import os

# Add the project root to sys.path so we can import the simulation module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from minimal_dual_network_simulation import create_minimal_network
from hnn_core import simulate_dipole


def test_minimal_network_drives_and_gids():
    # Create first minimal network
    net1 = create_minimal_network(cell_type_suffix=None, gid_start=0)
    next_gid = net1.get_next_gid()
    net1.add_evoked_drive(
        "evdist1",
        mu=5.0,
        sigma=1.0,
        numspikes=1,
        location="distal",
        weights_ampa={"L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
        gid_start=next_gid,
    )

    # Create second minimal network with renamed cell types and shifted GIDs
    next_gid = net1.get_next_gid()
    net2 = create_minimal_network(cell_type_suffix="_net2", gid_start=next_gid)
    next_gid_net2 = net2.get_next_gid()
    net2.add_evoked_drive(
        "evdist2",
        mu=5.0,
        sigma=1.0,
        numspikes=1,
        location="distal",
        weights_ampa={"L2_pyramidal_net2": 0.1, "L5_pyramidal_net2": 0.1},
        gid_start=next_gid_net2,
    )
    net2.dipole_cell_types = ["L2_pyramidal_net2", "L5_pyramidal_net2"]

    # Simulate both networks
    dpl_1 = simulate_dipole(net1, tstop=20, dt=0.025, n_trials=1)
    dpl_2 = simulate_dipole(net2, tstop=20, dt=0.025, n_trials=1)

    # Check that dipole output is not empty and has expected shape
    assert len(dpl_1) == 1
    assert len(dpl_2) == 1
    assert np.any(dpl_1[0].data["agg"])
    assert np.any(dpl_2[0].data["agg"])

    # Check that at least one spike occurred in each network
    assert len(net1.cell_response.spike_times[0]) > 0
    assert len(net2.cell_response.spike_times[0]) > 0

    # Check that cell type renaming and GID shifting worked
    for ct in ["L2_pyramidal", "L5_pyramidal"]:
        assert ct in net1.gid_ranges
    for ct in ["L2_pyramidal_net2", "L5_pyramidal_net2"]:
        assert ct in net2.gid_ranges
    # Ensure GID ranges do not overlap
    gids1 = set(gid for rng in net1.gid_ranges.values() for gid in rng)
    gids2 = set(gid for rng in net2.gid_ranges.values() for gid in rng)
    assert gids1.isdisjoint(gids2)

    # Check that drives are present
    assert "evdist1" in net1.external_drives
    assert "evdist2" in net2.external_drives
