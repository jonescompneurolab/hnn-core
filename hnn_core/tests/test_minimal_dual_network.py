import os
import sys
import numpy as np
import pytest

# Allow importing minimal_dual_network_simulation.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from minimal_dual_network_simulation import (
    create_minimal_network,
)  # assumes this exists
from hnn_core import simulate_dipole

TSTOP = 20.0
DT = 0.025


def _ensure_dipole_cts(net):
    # If you suffix cell types, ensure dipole list matches present pyramidal types
    if not hasattr(net, "dipole_cell_types") or not net.dipole_cell_types:
        cts = [k for k in net.gid_ranges.keys() if "pyramidal" in k]
        net.dipole_cell_types = cts


def _dipole_amp(dpl):
    arr = dpl[0].data["agg"]
    return float(np.max(arr) - np.min(arr))


def _gid_set(net):
    return set(g for rng in net.gid_ranges.values() for g in rng)


def test_minimal_network_drives_and_gids():
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

    next_gid2 = net1.get_next_gid()
    net2 = create_minimal_network(cell_type_suffix="_net2", gid_start=next_gid2)
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

    _ensure_dipole_cts(net1)
    _ensure_dipole_cts(net2)

    dpl1 = simulate_dipole(net1, tstop=TSTOP, dt=DT, n_trials=1)
    dpl2 = simulate_dipole(net2, tstop=TSTOP, dt=DT, n_trials=1)

    assert len(dpl1) == 1 and len(dpl2) == 1
    assert np.any(dpl1[0].data["agg"])
    assert np.any(dpl2[0].data["agg"])
    assert len(net1.cell_response.spike_times[0]) > 0
    assert len(net2.cell_response.spike_times[0]) > 0

    for ct in ["L2_pyramidal", "L5_pyramidal"]:
        assert ct in net1.gid_ranges
    for ct in ["L2_pyramidal_net2", "L5_pyramidal_net2"]:
        assert ct in net2.gid_ranges

    assert _gid_set(net1).isdisjoint(_gid_set(net2))
    assert "evdist1" in net1.external_drives
    assert "evdist2" in net2.external_drives


def test_evoked_weight_scaling():
    # Low weights
    net_low = create_minimal_network(None, gid_start=0)
    gid_low = net_low.get_next_gid()
    net_low.add_evoked_drive(
        "ev_low",
        mu=5.0,
        sigma=0.5,
        numspikes=1,
        location="distal",
        weights_ampa={"L2_pyramidal": 0.02, "L5_pyramidal": 0.02},
        gid_start=gid_low,
    )
    _ensure_dipole_cts(net_low)
    dpl_low = simulate_dipole(net_low, tstop=TSTOP, dt=DT, n_trials=1)
    amp_low = _dipole_amp(dpl_low)

    # High weights
    net_high = create_minimal_network(None, gid_start=0)
    gid_high = net_high.get_next_gid()
    net_high.add_evoked_drive(
        "ev_high",
        mu=5.0,
        sigma=0.5,
        numspikes=1,
        location="distal",
        weights_ampa={"L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
        gid_start=gid_high,
    )
    _ensure_dipole_cts(net_high)
    dpl_high = simulate_dipole(net_high, tstop=TSTOP, dt=DT, n_trials=1)
    amp_high = _dipole_amp(dpl_high)

    assert amp_low > 0
    assert amp_high > 0
    assert amp_high > amp_low


@pytest.mark.parametrize(
    "drive_cfg",
    [
        {
            "name": "evoked_distal",
            "api": "evoked",
            "location": "distal",
            "mu": 5.0,
            "sigma": 1.0,
            "numspikes": 1,
            "weights_ampa_base": {"L2_pyramidal": 0.05, "L5_pyramidal": 0.05},
            "event_seed": 11,
        },
        {
            "name": "evoked_proximal",
            "api": "evoked",
            "location": "proximal",
            "mu": 5.0,
            "sigma": 1.0,
            "numspikes": 1,
            "weights_ampa_base": {"L2_pyramidal": 0.05, "L5_pyramidal": 0.05},
            "event_seed": 12,
        },
        {
            "name": "poisson_proximal",
            "api": "poisson",
            "location": "proximal",
            "rate_constant": 8.0,
            "weights_ampa_base": {"L2_pyramidal": 0.03, "L5_pyramidal": 0.03},
            "event_seed": 21,
            "tstart": 2.0,
            "tstop": 15.0,
        },
        {
            "name": "poisson_distal",
            "api": "poisson",
            "location": "distal",
            "rate_constant": 8.0,
            "weights_ampa_base": {"L2_pyramidal": 0.03, "L5_pyramidal": 0.03},
            "event_seed": 22,
            "tstart": 2.0,
            "tstop": 15.0,
        },
        {
            "name": "bursty_distal",
            "api": "bursty",
            "location": "distal",
            "burst_rate": 55.0,
            "numspikes": 2,
            "spike_isi": 10.0,
            "weights_ampa_base": {"L2_pyramidal": 0.04, "L5_pyramidal": 0.04},
            "event_seed": 33,
            "tstart": 2.0,
            "tstop": 15.0,
        },
    ],
    ids=lambda cfg: cfg["name"],
)
def test_same_drive_type_two_networks(drive_cfg):
    net1 = create_minimal_network(None, gid_start=0)
    gid1 = net1.get_next_gid()
    # Create second network with suffixed cell type names
    net2 = create_minimal_network("_net2", gid_start=gid1)  # non-overlapping start
    gid2 = net2.get_next_gid()
    net2.dipole_cell_types = [ct for ct in net2.gid_ranges if "pyramidal" in ct]

    # Prepare weight dicts
    w1 = drive_cfg["weights_ampa_base"]
    w2 = {f"{k}_net2": v for k, v in w1.items()}

    api = drive_cfg["api"]
    if api == "evoked":
        net1.add_evoked_drive(
            drive_cfg["name"] + "_n1",
            mu=drive_cfg["mu"],
            sigma=drive_cfg["sigma"],
            numspikes=drive_cfg["numspikes"],
            location=drive_cfg["location"],
            weights_ampa=w1,
            event_seed=drive_cfg["event_seed"],
            gid_start=gid1,
        )
        net2.add_evoked_drive(
            drive_cfg["name"] + "_n2",
            mu=drive_cfg["mu"],
            sigma=drive_cfg["sigma"],
            numspikes=drive_cfg["numspikes"],
            location=drive_cfg["location"],
            weights_ampa=w2,
            event_seed=drive_cfg["event_seed"],
            gid_start=gid2,
        )
    elif api == "poisson":
        net1.add_poisson_drive(
            drive_cfg["name"] + "_n1",
            tstart=drive_cfg.get("tstart", 0.0),
            tstop=drive_cfg.get("tstop", TSTOP - 1),
            rate_constant=drive_cfg["rate_constant"],
            location=drive_cfg["location"],
            weights_ampa=w1,
            event_seed=drive_cfg["event_seed"],
            gid_start=gid1,
        )
        net2.add_poisson_drive(
            drive_cfg["name"] + "_n2",
            tstart=drive_cfg.get("tstart", 0.0),
            tstop=drive_cfg.get("tstop", TSTOP - 1),
            rate_constant=drive_cfg["rate_constant"],
            location=drive_cfg["location"],
            weights_ampa=w2,
            event_seed=drive_cfg["event_seed"],
            gid_start=gid2,
        )
    elif api == "bursty":
        net1.add_bursty_drive(
            drive_cfg["name"] + "_n1",
            tstart=drive_cfg.get("tstart", 0.0),
            tstop=drive_cfg.get("tstop", TSTOP - 1),
            burst_rate=drive_cfg["burst_rate"],
            location=drive_cfg["location"],
            numspikes=drive_cfg["numspikes"],
            spike_isi=drive_cfg["spike_isi"],
            weights_ampa=w1,
            event_seed=drive_cfg["event_seed"],
            gid_start=gid1,
        )
        net2.add_bursty_drive(
            drive_cfg["name"] + "_n2",
            tstart=drive_cfg.get("tstart", 0.0),
            tstop=drive_cfg.get("tstop", TSTOP - 1),
            burst_rate=drive_cfg["burst_rate"],
            location=drive_cfg["location"],
            numspikes=drive_cfg["numspikes"],
            spike_isi=drive_cfg["spike_isi"],
            weights_ampa=w2,
            event_seed=drive_cfg["event_seed"],
            gid_start=gid2,
        )
    else:
        pytest.skip(f"Unsupported drive api {api}")

    _ensure_dipole_cts(net1)
    _ensure_dipole_cts(net2)

    dpl1 = simulate_dipole(net1, tstop=TSTOP, dt=DT, n_trials=1)
    dpl2 = simulate_dipole(net2, tstop=TSTOP, dt=DT, n_trials=1)
    amp1 = _dipole_amp(dpl1)
    amp2 = _dipole_amp(dpl2)

    assert amp1 > 0 and amp2 > 0
    # Loose similarity check (stochastic differences allowed)
    assert abs(amp1 - amp2) / max(amp1, amp2) < 0.9
