# Authors: Mainak Jas <mainakjas@gmail.com>

import os.path as op
import numpy as np
import pytest

import hnn_core
from hnn_core import read_params, jones_2009_model, simulate_dipole
from hnn_core.optimization.optimize_evoked import (
    _consolidate_chunks,
    _split_by_evinput,
    _generate_weights,
    _get_drive_params,
    optimize_evoked,
)


def test_consolidate_chunks():
    """Test consolidation of chunks."""
    inputs = {
        "ev1": {
            "start": 5,
            "end": 25,
            "ranges": {"initial": 1e-10, "minval": 1e-11, "maxval": 1e-9},
            "opt_end": 90,
            "weights": np.array([5.0, 10.0]),
        },
        "ev2": {
            "start": 100,
            "end": 120,
            "ranges": {"initial": 1e-10, "minval": 1e-11, "maxval": 1e-9},
            "opt_end": 170,
            "weights": np.array([10.0, 5.0]),
        },
    }
    chunks = _consolidate_chunks(inputs)
    assert len(chunks) == len(inputs) + 1  # extra last chunk??
    assert chunks[-1]["opt_end"] == inputs["ev2"]["opt_end"]
    assert chunks[-1]["inputs"] == ["ev1", "ev2"]
    assert isinstance(chunks, list)

    # overlapping chunks
    inputs["ev1"]["end"] = 110
    chunks = _consolidate_chunks(inputs)
    assert len(chunks) == 1
    assert chunks[0]["start"] == inputs["ev1"]["start"]
    assert chunks[0]["end"] == inputs["ev2"]["end"]
    assert np.allclose(
        chunks[0]["weights"],
        (inputs["ev1"]["weights"] + inputs["ev2"]["weights"]) / 2.0,
    )


def test_split_by_evinput():
    """Test splitting evoked input."""
    drive_names = ["ev_drive_1", "ev_drive_2"]
    drive_dynamics = [{"mu": 5.0, "sigma": 0.1}, {"mu": 10.0, "sigma": 0.2}]
    drive_syn_weights = [{"ampa_L2_pyramidal": 1.0}, {"nmda_L5_basket": 2.0}]
    tstop = 20.0
    dt = 0.025

    timing_range_multiplier = 3.0
    sigma_range_multiplier = 50.0
    synweight_range_multiplier = 500.0
    decay_multiplier = 1.6
    evinput_params = _split_by_evinput(
        drive_names,
        drive_dynamics,
        drive_syn_weights,
        tstop,
        sigma_range_multiplier,
        timing_range_multiplier,
        synweight_range_multiplier,
    )
    assert list(evinput_params.keys()) == drive_names
    for evinput in evinput_params.values():
        assert list(evinput.keys()) == ["mean", "sigma", "ranges", "start", "end"]

    evinput_params = _generate_weights(evinput_params, tstop, dt, decay_multiplier)
    for evinput in evinput_params.values():
        assert list(evinput.keys()) == [
            "ranges",
            "start",
            "end",
            "weights",
            "opt_start",
            "opt_end",
        ]


def test_optimize_evoked():
    """Test running the full routine in a reduced network."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)

    tstop = 10.0
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    mu_orig = 6.0
    params.update(
        {
            "t_evprox_1": mu_orig,
            "sigma_t_evprox_1": 2.0,
            "t_evdist_1": mu_orig + 2,
            "sigma_t_evdist_1": 2.0,
        }
    )
    net_orig = jones_2009_model(params, add_drives_from_params=True, mesh_shape=(3, 3))
    del net_orig.external_drives["evprox2"]
    dpl_orig = simulate_dipole(net_orig, tstop=tstop, n_trials=n_trials)[0]

    # simulate a dipole with a time-shifted drive
    mu_offset = 4.0
    params.update(
        {
            "t_evprox_1": mu_offset,
            "sigma_t_evprox_1": 2.0,
            "t_evdist_1": mu_offset + 2,
            "sigma_t_evdist_1": 2.0,
        }
    )
    net_offset = jones_2009_model(
        params, add_drives_from_params=True, mesh_shape=(3, 3)
    )
    del net_offset.external_drives["evprox2"]
    dpl_offset = simulate_dipole(net_offset, tstop=tstop, n_trials=n_trials)[0]
    # get drive params from the pre-optimization Network instance
    _, _, drive_static_params_orig = _get_drive_params(net_offset, ["evprox1"])

    with pytest.raises(
        ValueError, match="The current Network instance lacks any evoked drives"
    ):
        net_empty = net_offset.copy()
        del net_empty.external_drives["evprox1"]
        del net_empty.external_drives["evdist1"]
        net_opt = optimize_evoked(
            net_empty,
            tstop=tstop,
            n_trials=n_trials,
            target_dpl=dpl_orig,
            initial_dpl=dpl_offset,
            maxiter=10,
        )

    with pytest.raises(
        ValueError, match="The drives selected to be optimized are not evoked drives"
    ):
        net_test_bursty = net_offset.copy()
        which_drives = ["bursty1"]
        net_opt = optimize_evoked(
            net_test_bursty,
            tstop=tstop,
            n_trials=n_trials,
            target_dpl=dpl_orig,
            initial_dpl=dpl_offset,
            which_drives=which_drives,
            maxiter=10,
        )

    which_drives = ["evprox1"]  # drive selected to optimize
    maxiter = 12
    # try without returning iteration RMSE first
    net_opt = optimize_evoked(
        net_offset,
        tstop=tstop,
        n_trials=n_trials,
        target_dpl=dpl_orig,
        initial_dpl=dpl_offset,
        timing_range_multiplier=3.0,
        sigma_range_multiplier=50.0,
        synweight_range_multiplier=500.0,
        maxiter=maxiter,
        which_drives=which_drives,
        return_rmse=False,
    )
    net_opt, rmse = optimize_evoked(
        net_offset,
        tstop=tstop,
        n_trials=n_trials,
        target_dpl=dpl_orig,
        initial_dpl=dpl_offset,
        timing_range_multiplier=3.0,
        sigma_range_multiplier=50.0,
        synweight_range_multiplier=500.0,
        maxiter=maxiter,
        which_drives=which_drives,
        return_rmse=True,
    )

    # the number of returned rmse values should be the same as maxiter
    assert len(rmse) <= maxiter

    # the returned rmse values should be positive
    assert all(vals > 0 for vals in rmse)

    # the names of drives should be preserved during optimization
    assert net_offset.external_drives.keys() == net_opt.external_drives.keys()

    drive_dynamics_opt, drive_syn_weights_opt, drive_static_params_opt = (
        _get_drive_params(net_opt, ["evprox1"])
    )

    # ensure that params corresponding to only one evoked drive are discovered
    assert (
        len(drive_dynamics_opt)
        == len(drive_syn_weights_opt)
        == len(drive_static_params_opt)
        == 1
    )

    # static drive params should remain constant
    assert drive_static_params_opt == drive_static_params_orig

    # ensure that only the drive that we wanted to optimize over changed
    (
        drive_evdist1_dynamics_offset,
        drive_evdist1_syn_weights_offset,
        drive_static_params_offset,
    ) = _get_drive_params(net_offset, ["evdist1"])
    (
        drive_evdist1_dynamics_opt,
        drive_evdist1_syn_weights_opt,
        drive_static_params_opt,
    ) = _get_drive_params(net_opt, ["evdist1"])

    # assert that evdist1 did NOT change
    assert drive_evdist1_dynamics_opt == drive_evdist1_dynamics_offset
    assert drive_evdist1_syn_weights_opt == drive_evdist1_syn_weights_offset
