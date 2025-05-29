# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import pytest
import os.path as op

import numpy as np

import hnn_core
from hnn_core import Network, read_params
from hnn_core.drives import (
    _drive_cell_event_times,
    _get_prng,
    _create_extpois,
    _create_bursty_input,
)
from hnn_core.network import pick_connection
from hnn_core.network_models import jones_2009_model
from hnn_core import simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)


@pytest.fixture
def setup_net():
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)
    net = jones_2009_model(params, mesh_shape=(3, 3))

    return net


def test_external_drive_times():
    """Test the different external drives."""

    drive_type = "invalid_drive"
    dynamics = dict(mu=5, sigma=0.5, numspikes=1)
    tstop = 10
    pytest.raises(ValueError, _drive_cell_event_times, "invalid_drive", dynamics, tstop)
    pytest.raises(
        ValueError, _drive_cell_event_times, "ss", dynamics, tstop
    )  # ambiguous

    # validate poisson input time interval
    drive_type = "poisson"
    dynamics = {
        "tstart": 0,
        "tstop": 250.0,
        "rate_constant": {
            "L2_basket": 1,
            "L2_pyramidal": 140.0,
            "L5_basket": 1,
            "L5_pyramidal": 40.0,
        },
    }
    with pytest.raises(ValueError, match="The end time for Poisson input"):
        dynamics["tstop"] = -1
        event_times = _drive_cell_event_times(
            drive_type=drive_type, dynamics=dynamics, tstop=tstop
        )
    with pytest.raises(ValueError, match="The start time for Poisson"):
        dynamics["tstop"] = tstop
        dynamics["tstart"] = -1
        event_times = _drive_cell_event_times(
            drive_type=drive_type, dynamics=dynamics, tstop=tstop
        )

    # checks the poisson spike train generation
    prng = np.random.RandomState()
    lamtha = 50.0
    event_times = _create_extpois(t0=0, T=100000, lamtha=lamtha, prng=prng)
    event_intervals = np.diff(event_times)
    assert pytest.approx(event_intervals.mean(), abs=1.5) == 1000 * 1 / lamtha

    with pytest.raises(ValueError, match="The start time for Poisson"):
        _create_extpois(t0=-5, T=5, lamtha=lamtha, prng=prng)
    with pytest.raises(ValueError, match="The end time for Poisson"):
        _create_extpois(t0=50, T=20, lamtha=lamtha, prng=prng)
    with pytest.raises(ValueError, match="Rate must be > 0"):
        _create_extpois(t0=0, T=1000, lamtha=-5, prng=prng)

    # check bursty/rhythmic input
    t0 = 0
    t0_stdev = 5
    tstop = 100
    f_input = 20.0
    events_per_cycle = 3
    cycle_events_isi = 7
    events_jitter_std = 5.0
    prng, prng2 = _get_prng(seed=0, gid=5)
    event_times = _create_bursty_input(
        t0=t0,
        t0_stdev=t0_stdev,
        tstop=tstop,
        f_input=f_input,
        events_jitter_std=events_jitter_std,
        events_per_cycle=events_per_cycle,
        cycle_events_isi=cycle_events_isi,
        prng=prng,
        prng2=prng2,
    )

    events_per_cycle = 5
    cycle_events_isi = 20
    with pytest.raises(
        ValueError, match=r"(?s)Burst duration .* cannot be greater than"
    ):
        _create_bursty_input(
            t0=t0,
            t0_stdev=t0_stdev,
            tstop=tstop,
            f_input=f_input,
            events_jitter_std=events_jitter_std,
            events_per_cycle=events_per_cycle,
            cycle_events_isi=cycle_events_isi,
            prng=prng,
            prng2=prng2,
        )


def test_drive_seeds(setup_net):
    """Test that unique spike times are generated across trials"""
    net = setup_net
    weights_ampa = {
        "L2_basket": 0.3,
        "L2_pyramidal": 0.3,
        "L5_basket": 0.3,
        "L5_pyramidal": 0.3,
    }
    synaptic_delays = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    net.add_evoked_drive(
        "prox",
        mu=40,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="proximal",
        synaptic_delays=synaptic_delays,
        event_seed=1,
    )

    _ = simulate_dipole(net, tstop=100, dt=0.5, n_trials=2)
    trial1_spikes = np.array(sorted(net.external_drives["prox"]["events"][0]))
    trial2_spikes = np.array(sorted(net.external_drives["prox"]["events"][1]))
    # No two spikes should be perfectly identical across seeds
    assert ~np.any(np.allclose(trial1_spikes, trial2_spikes))


def test_clear_drives(setup_net):
    """Test clearing drives updates Network"""
    net = setup_net
    weights_ampa = {"L5_pyramidal": 0.3}
    synaptic_delays = {"L5_pyramidal": 1.0}

    # Test attributes after adding 2 drives
    n_gids = net._n_gids
    net.add_evoked_drive(
        "prox",
        mu=40,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="proximal",
        synaptic_delays=synaptic_delays,
        cell_specific=True,
    )

    net.add_evoked_drive(
        "dist",
        mu=40,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="distal",
        synaptic_delays=synaptic_delays,
        cell_specific=True,
    )

    for drive_name in ["prox", "dist"]:
        assert len(net.external_drives) == 2
        assert drive_name in net.external_drives
        assert drive_name in net.gid_ranges
        assert drive_name in net.pos_dict
        assert net._n_gids == n_gids + len(net.gid_ranges["L5_pyramidal"]) * 2

    # Test attributes after clearing drives
    net.clear_drives()
    for drive_name in ["prox", "dist"]:
        assert len(net.external_drives) == 0
        assert drive_name not in net.external_drives
        assert drive_name not in net.gid_ranges
        assert drive_name not in net.pos_dict
        assert net._n_gids == n_gids

    # Test attributes after adding 1 drive
    net.add_evoked_drive(
        "prox",
        mu=40,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="proximal",
        synaptic_delays=synaptic_delays,
        cell_specific=True,
    )

    assert len(net.external_drives) == 1
    assert "prox" in net.external_drives
    assert "prox" in net.gid_ranges
    assert "prox" in net.pos_dict
    assert net._n_gids == n_gids + len(net.gid_ranges["L5_pyramidal"])


def test_add_drives():
    """Test methods for adding drives to a Network."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)
    net = Network(params, legacy_mode=False)

    # Ensure weights and delays are updated
    weights_ampa = {"L2_basket": 1.0, "L2_pyramidal": 3.0, "L5_pyramidal": 4.0}
    syn_delays = {"L2_basket": 1.0, "L2_pyramidal": 2.0, "L5_pyramidal": 4.0}

    n_drive_cells = 10
    cell_specific = False  # default for bursty drive
    net.add_bursty_drive(
        "bursty",
        location="distal",
        burst_rate=10,
        weights_ampa=weights_ampa,
        synaptic_delays=syn_delays,
        n_drive_cells=n_drive_cells,
    )

    assert net.external_drives["bursty"]["n_drive_cells"] == n_drive_cells
    assert net.external_drives["bursty"]["cell_specific"] == cell_specific
    conn_idxs = pick_connection(net, src_gids="bursty")
    for conn_idx in conn_idxs:
        drive_conn = net.connectivity[conn_idx]
        target_type = drive_conn["target_type"]
        assert drive_conn["nc_dict"]["A_weight"] == weights_ampa[target_type]
        assert drive_conn["nc_dict"]["A_delay"] == syn_delays[target_type]

    n_drive_cells = "n_cells"  # default for evoked drive
    cell_specific = True
    net.add_evoked_drive(
        "evoked_dist",
        mu=1.0,
        sigma=1.0,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="distal",
        synaptic_delays=syn_delays,
        cell_specific=True,
    )

    n_dist_targets = 235  # 270 with legacy mode
    assert net.external_drives["evoked_dist"]["n_drive_cells"] == n_dist_targets
    assert net.external_drives["evoked_dist"]["cell_specific"] == cell_specific
    conn_idxs = pick_connection(net, src_gids="evoked_dist")
    for conn_idx in conn_idxs:
        drive_conn = net.connectivity[conn_idx]
        target_type = drive_conn["target_type"]
        assert drive_conn["nc_dict"]["A_weight"] == weights_ampa[target_type]
        assert drive_conn["nc_dict"]["A_delay"] == syn_delays[target_type]

    n_drive_cells = "n_cells"  # default for poisson drive
    cell_specific = True
    net.add_poisson_drive(
        "poisson",
        rate_constant=1.0,
        weights_ampa=weights_ampa,
        location="distal",
        synaptic_delays=syn_delays,
        cell_specific=cell_specific,
    )

    n_dist_targets = 235  # 270 with non-legacy mode
    assert net.external_drives["poisson"]["n_drive_cells"] == n_dist_targets
    assert net.external_drives["poisson"]["cell_specific"] == cell_specific
    conn_idxs = pick_connection(net, src_gids="poisson")
    for conn_idx in conn_idxs:
        drive_conn = net.connectivity[conn_idx]
        target_type = drive_conn["target_type"]
        assert drive_conn["nc_dict"]["A_weight"] == weights_ampa[target_type]
        assert drive_conn["nc_dict"]["A_delay"] == syn_delays[target_type]

    # Test drive targeting specific section
    # Section present on all cells indicated
    location = "apical_tuft"
    weights_ampa_tuft = {"L2_pyramidal": 1.0, "L5_pyramidal": 2.0}
    syn_delays_tuft = {"L2_pyramidal": 1.0, "L5_pyramidal": 2.0}
    net.add_bursty_drive(
        "bursty_tuft",
        location=location,
        burst_rate=10,
        weights_ampa=weights_ampa_tuft,
        synaptic_delays=syn_delays_tuft,
        n_drive_cells=10,
    )
    assert net.connectivity[-1]["loc"] == location

    # Section not present on cells indicated
    location = "apical_tuft"
    weights_ampa_no_tuft = {"L2_pyramidal": 1.0, "L5_basket": 2.0}
    syn_delays_no_tuft = {"L2_pyramidal": 1.0, "L5_basket": 2.0}
    match = "Invalid value for"
    with pytest.raises(ValueError, match=match):
        net.add_bursty_drive(
            "bursty_no_tuft",
            location=location,
            burst_rate=10,
            weights_ampa=weights_ampa_no_tuft,
            synaptic_delays=syn_delays_no_tuft,
            n_drive_cells=n_drive_cells,
        )

    # Test probabilistic drive connections.
    # drive with cell_specific=False
    n_drive_cells = 10
    probability = 0.5  # test that only half of possible connections are made
    weights_nmda = {"L2_basket": 1.0, "L2_pyramidal": 3.0, "L5_pyramidal": 4.0}
    net.add_bursty_drive(
        "bursty_prob",
        location="distal",
        burst_rate=10,
        weights_ampa=weights_ampa,
        weights_nmda=weights_nmda,
        synaptic_delays=syn_delays,
        n_drive_cells=n_drive_cells,
        probability=probability,
    )

    for cell_type in weights_ampa.keys():
        conn_idxs = pick_connection(net, src_gids="bursty_prob", target_gids=cell_type)
        gid_pairs_comparison = net.connectivity[conn_idxs[0]]["gid_pairs"]
        for conn_idx in conn_idxs:
            conn = net.connectivity[conn_idx]
            num_connections = np.sum([len(gids) for gids in conn["gid_pairs"].values()])
            # Ensures that AMPA and NMDA connections target the same gids.
            # Necessary when weights of both are non-zero.
            assert gid_pairs_comparison == conn["gid_pairs"]
            assert num_connections == np.around(
                len(net.gid_ranges[cell_type]) * n_drive_cells * probability
            ).astype(int)

    # drives with cell_specific=True
    probability = {"L2_basket": 0.1, "L2_pyramidal": 0.25, "L5_pyramidal": 0.5}
    net.add_evoked_drive(
        "evoked_prob",
        mu=1.0,
        sigma=1.0,
        numspikes=1,
        weights_ampa=weights_ampa,
        weights_nmda=weights_nmda,
        location="distal",
        synaptic_delays=syn_delays,
        cell_specific=True,
        probability=probability,
    )

    for cell_type in weights_ampa.keys():
        conn_idxs = pick_connection(net, src_gids="evoked_prob", target_gids=cell_type)
        gid_pairs_comparison = net.connectivity[conn_idxs[0]]["gid_pairs"]
        for conn_idx in conn_idxs:
            conn = net.connectivity[conn_idx]
            num_connections = np.sum([len(gids) for gids in conn["gid_pairs"].values()])
            assert gid_pairs_comparison == conn["gid_pairs"]
            assert num_connections == np.around(
                len(net.gid_ranges[cell_type]) * probability[cell_type]
            ).astype(int)

    # Test adding just the NMDA weights (no AMPA)
    net.add_evoked_drive(
        "evoked_nmda",
        mu=1.0,
        sigma=1.0,
        numspikes=1,
        weights_nmda=weights_nmda,
        location="distal",
        synaptic_delays=syn_delays,
        cell_specific=True,
        probability=probability,
    )

    # Round trip test to ensure drives API produces a functioning Network
    simulate_dipole(net, tstop=1)

    # evoked
    with pytest.raises(ValueError, match="Standard deviation cannot be negative"):
        net.add_evoked_drive("evdist1", mu=10, sigma=-1, numspikes=1, location="distal")
    with pytest.raises(ValueError, match="Number of spikes must be greater than zero"):
        net.add_evoked_drive("evdist1", mu=10, sigma=1, numspikes=0, location="distal")

    # Test Network._attach_drive()
    with pytest.raises(ValueError, match="Invalid value for"):
        net.add_evoked_drive(
            "evdist1",
            mu=10,
            sigma=1,
            numspikes=1,
            location="bogus_location",
            weights_ampa={"L5_basket": 1.0},
            synaptic_delays={"L5_basket": 0.1},
        )
    with pytest.raises(ValueError, match="Drive evoked_dist already defined"):
        net.add_evoked_drive(
            "evoked_dist", mu=10, sigma=1, numspikes=1, location="distal"
        )
    with pytest.raises(
        ValueError, match="No target cell types have been given a synaptic weight"
    ):
        net.add_evoked_drive("evdist1", mu=10, sigma=1, numspikes=1, location="distal")
    with pytest.raises(
        ValueError,
        match="Due to physiological/anatomical constraints, "
        "a distal drive cannot target L5_basket cell types. ",
    ):
        net.add_evoked_drive(
            "evdist1",
            mu=10,
            sigma=1,
            numspikes=1,
            location="distal",
            weights_ampa={"L5_basket": 1.0},
            synaptic_delays={"L5_basket": 0.1},
        )
    with pytest.raises(ValueError, match="If cell_specific is True, n_drive_cells"):
        net.add_evoked_drive(
            "evdist1",
            mu=10,
            sigma=1,
            numspikes=1,
            location="distal",
            n_drive_cells=10,
            cell_specific=True,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )
    with pytest.raises(ValueError, match="If cell_specific is False, n_drive_cells"):
        net.add_evoked_drive(
            "evdist1",
            mu=10,
            sigma=1,
            numspikes=1,
            location="distal",
            n_drive_cells="n_cells",
            cell_specific=False,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )
    with pytest.raises(
        ValueError, match="Number of drive cells must be greater than 0"
    ):
        net.add_evoked_drive(
            "evdist1",
            mu=10,
            sigma=1,
            numspikes=1,
            location="distal",
            n_drive_cells=0,
            cell_specific=False,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )

    # Poisson
    with pytest.raises(
        ValueError, match="End time of Poisson drive cannot be negative"
    ):
        net.add_poisson_drive(
            "poisson1", tstart=0, tstop=-1, location="distal", rate_constant=10.0
        )
    with pytest.raises(
        ValueError, match="Start time of Poisson drive cannot be negative"
    ):
        net.add_poisson_drive(
            "poisson1", tstart=-1, location="distal", rate_constant=10.0
        )
    with pytest.raises(
        ValueError, match="Duration of Poisson drive cannot be negative"
    ):
        net.add_poisson_drive(
            "poisson1", tstart=10, tstop=1, location="distal", rate_constant=10.0
        )
    with pytest.raises(ValueError, match="Rate constant must be positive"):
        net.add_poisson_drive(
            "poisson1",
            location="distal",
            rate_constant=0.0,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )

    with pytest.raises(ValueError, match="Rate constants not provided for all target"):
        net.add_poisson_drive(
            "poisson1",
            location="distal",
            rate_constant={"L2_pyramidal": 10.0},
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )
    with pytest.raises(
        ValueError, match="Rate constant provided for unknown target cell"
    ):
        net.add_poisson_drive(
            "poisson1",
            location="distal",
            rate_constant={"L2_pyramidal": 10.0, "bogus_celltype": 20.0},
            weights_ampa={"L2_pyramidal": 0.01, "bogus_celltype": 0.01},
            synaptic_delays=0.1,
        )

    with pytest.raises(
        ValueError,
        match="Drives specific to cell types are only possible with cell_specific=True",
    ):
        net.add_poisson_drive(
            "poisson1",
            location="distal",
            rate_constant={
                "L2_basket": 10.0,
                "L2_pyramidal": 11.0,
                "L5_basket": 12.0,
                "L5_pyramidal": 13.0,
            },
            n_drive_cells=1,
            cell_specific=False,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )

    # bursty
    with pytest.raises(ValueError, match="End time of bursty drive cannot be negative"):
        net.add_bursty_drive("bursty_drive", tstop=-1, location="distal", burst_rate=10)
    with pytest.raises(
        ValueError, match="Start time of bursty drive cannot be negative"
    ):
        net.add_bursty_drive(
            "bursty_drive", tstart=-1, location="distal", burst_rate=10
        )
    with pytest.raises(ValueError, match="Duration of bursty drive cannot be negative"):
        net.add_bursty_drive(
            "bursty_drive", tstart=10, tstop=1, location="distal", burst_rate=10
        )

    msg = (
        r"(?s)Burst duration .* cannot be greater than "
        "burst period"
    )
    with pytest.raises(ValueError, match=msg):
        net.add_bursty_drive(
            "bursty_drive",
            location="distal",
            burst_rate=10,
            burst_std=20.0,
            numspikes=4,
            spike_isi=50,
        )

    # attaching drives
    with pytest.raises(ValueError, match="Drive evoked_dist already defined"):
        net.add_poisson_drive(
            "evoked_dist",
            location="distal",
            rate_constant=10.0,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )
    with pytest.raises(ValueError, match="Invalid value for the"):
        net.add_poisson_drive(
            "weird_poisson",
            location="between",
            rate_constant=10.0,
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
        )
    with pytest.raises(ValueError, match="Allowed drive target cell types are:"):
        net.add_poisson_drive(
            "cell_unknown",
            location="proximal",
            rate_constant=10.0,
            weights_ampa={"CA1_pyramidal": 1.0},
            synaptic_delays=0.01,
        )
    with pytest.raises(
        ValueError,
        match="synaptic_delays is either a common float or "
        "needs to be specified as a dict for each of the cell",
    ):
        net.add_poisson_drive(
            "cell_unknown",
            location="proximal",
            rate_constant=10.0,
            weights_ampa={"L2_pyramidal": 1.0},
            synaptic_delays={"L5_pyramidal": 1.0},
        )
    with pytest.raises(ValueError, match=r"probability must be in the range \(0\,1\)"):
        net.add_bursty_drive(
            "cell_unknown",
            location="distal",
            burst_rate=10,
            weights_ampa={"L2_pyramidal": 1.0},
            synaptic_delays={"L2_pyramidal": 1.0},
            probability=2.0,
        )

    with pytest.raises(
        TypeError,
        match="probability must be an instance of "
        r"float or dict, got \<class 'str'\> instead",
    ):
        net.add_bursty_drive(
            "cell_unknown2",
            location="distal",
            burst_rate=10,
            weights_ampa={"L2_pyramidal": 1.0},
            synaptic_delays={"L2_pyramidal": 1.0},
            probability="1.0",
        )

    with pytest.raises(
        ValueError,
        match="probability is either a common "
        "float or needs to be specified as a dict for "
        "each of the cell",
    ):
        net.add_bursty_drive(
            "cell_unknown2",
            location="distal",
            burst_rate=10,
            weights_ampa={"L2_pyramidal": 1.0},
            synaptic_delays={"L2_pyramidal": 1.0},
            probability={"L5_pyramidal": 1.0},
        )

    with pytest.raises(
        TypeError,
        match="probability must be an instance of "
        r"float, got \<class 'str'\> instead",
    ):
        net.add_bursty_drive(
            "cell_unknown2",
            location="distal",
            burst_rate=10,
            weights_ampa={"L2_pyramidal": 1.0},
            synaptic_delays={"L2_pyramidal": 1.0},
            probability={"L2_pyramidal": "1.0"},
        )

    with pytest.raises(ValueError, match=r"probability must be in the range \(0\,1\)"):
        net.add_bursty_drive(
            "cell_unknown3",
            location="distal",
            burst_rate=10,
            weights_ampa={"L2_pyramidal": 1.0},
            synaptic_delays={"L2_pyramidal": 1.0},
            probability={"L2_pyramidal": 2.0},
        )

    with pytest.warns(UserWarning, match="No external drives or biases load"):
        net.clear_drives()
        simulate_dipole(net, tstop=10)


def test_drive_random_state():
    """Tests to check same random state always gives same spike times."""

    weights_ampa = {
        "L2_basket": 0.08,
        "L2_pyramidal": 0.02,
        "L5_basket": 0.2,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    net = jones_2009_model()
    for drive_name in ["evprox1", "evprox2"]:
        net.add_evoked_drive(
            drive_name,
            mu=137.12,
            sigma=8,
            numspikes=1,
            weights_ampa=weights_ampa,
            weights_nmda=None,
            location="proximal",
            synaptic_delays=synaptic_delays,
            event_seed=4,
        )

    net._instantiate_drives(tstop=170.0)
    assert (
        net.external_drives["evprox1"]["events"]
        == net.external_drives["evprox2"]["events"]
    )


@pytest.mark.parametrize(
    "rate_constant,cell_specific,n_drive_cells",
    [
        (2, False, 1),
        (2.0, False, 1),
        (2, True, "n_cells"),
        (2.0, True, "n_cells"),
    ],
)
def test_add_poisson_drive(setup_net, rate_constant, cell_specific, n_drive_cells):
    """Testing rate constant when adding non-cell-specific poisson drive"""
    net = setup_net

    weights_ampa_noise = {
        "L2_basket": 0.01,
        "L2_pyramidal": 0.002,
        "L5_pyramidal": 0.02,
    }

    net.add_poisson_drive(
        "noise_global",
        rate_constant=rate_constant,
        location="distal",
        weights_ampa=weights_ampa_noise,
        space_constant=100,
        n_drive_cells=n_drive_cells,
        cell_specific=cell_specific,
    )

    simulate_dipole(net, tstop=5)
