import os.path as op
from urllib.request import urlretrieve
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from h5io import write_hdf5
import pytest

import hnn_core
from hnn_core import read_params, read_dipole, average_dipoles
from hnn_core import Network, jones_2009_model
from hnn_core.viz import plot_dipole
from hnn_core.dipole import Dipole, simulate_dipole, _rmse
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil

matplotlib.use("agg")


def test_dipole(tmp_path, run_hnn_core_fixture):
    """Test dipole object."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, "param", "default.json")
    dpl_out_fname = tmp_path / "dpl1.txt"
    dpl_out_hdf5_fname = tmp_path / "dpl.hdf5"
    params = read_params(params_fname)
    times = np.arange(0, 6000 * params["dt"], params["dt"])
    data = np.random.random((6000, 3))
    dipole = Dipole(times, data)
    dipole._baseline_renormalize(params["N_pyr_x"], params["N_pyr_y"])
    dipole._convert_fAm_to_nAm()

    # test smoothing and scaling
    dipole_raw = dipole.copy()
    dipole.scale(params["dipole_scalefctr"])
    dipole.smooth(window_len=params["dipole_smooth_win"])
    with pytest.raises(AssertionError):
        assert_allclose(dipole.data["agg"], dipole_raw.data["agg"])
    assert_allclose(
        dipole.data["agg"],
        (
            params["dipole_scalefctr"]
            * dipole_raw.smooth(params["dipole_smooth_win"]).data["agg"]
        ),
    )

    dipole.plot(show=False)
    plot_dipole([dipole, dipole], show=False)

    # Test wrong argument to plot_dipole()
    with pytest.raises(TypeError, match="dpl must be an instance of"):
        plot_dipole([dipole, 10], show=False)
    with pytest.raises(
        AttributeError, match="'numpy.ndarray' object has no attribute 'append'"
    ):
        plot_dipole(np.array([dipole, dipole]), average=True, show=False)

    # Test IO for txt files
    dipole.write(dpl_out_fname)
    dipole_read = read_dipole(dpl_out_fname)
    assert_allclose(dipole_read.times, dipole.times, rtol=0, atol=0.00051)
    for dpl_key in dipole.data.keys():
        assert_allclose(
            dipole_read.data[dpl_key], dipole.data[dpl_key], rtol=0, atol=0.000051
        )

    # Test IO for hdf5 files
    dipole.write(dpl_out_hdf5_fname)
    dipole_read_hdf5 = read_dipole(dpl_out_hdf5_fname)
    assert_allclose(dipole_read_hdf5.times, dipole.times, rtol=0.00051, atol=0)
    for dpl_key in dipole.data.keys():
        assert_allclose(
            dipole_read_hdf5.data[dpl_key], dipole.data[dpl_key], rtol=0.000051, atol=0
        )

    # Test read for hdf5 files using Path object
    dipole_read_hdf5 = read_dipole(Path(dpl_out_hdf5_fname))
    assert_allclose(dipole_read_hdf5.times, dipole.times, rtol=0.00051, atol=0)
    for dpl_key in dipole.data.keys():
        assert_allclose(
            dipole_read_hdf5.data[dpl_key], dipole.data[dpl_key], rtol=0.000051, atol=0
        )

    # Testing when overwrite is False and same filename is used
    with pytest.raises(FileExistsError, match="File already exists at path "):
        dipole.write(dpl_out_hdf5_fname, overwrite=False)

    # Testing for wrong extension provided
    with pytest.raises(NameError, match="File extension should be either txt or hdf5"):
        dipole.write(tmp_path / "dpl.xls")

    # Testing File Not Found Error
    with pytest.raises(FileNotFoundError, match="File not found at "):
        read_dipole(tmp_path / "dpl1.hdf5")

    # dpls with different scale_applied should not be averaged.
    with pytest.raises(RuntimeError, match="All dipoles must be scaled equally"):
        dipole_avg = average_dipoles([dipole, dipole_read])
    # Checking object type field not exists error
    dummy_data = dict()
    dummy_data["objective"] = "Check Object type errors"
    write_hdf5(tmp_path / "not_dpl.hdf5", dummy_data)
    with pytest.raises(NameError, match="The given file is not compatible."):
        read_dipole(tmp_path / "not_dpl.hdf5")
    # Checking wrong object type error
    dummy_data["object_type"] = "dpl"
    write_hdf5(tmp_path / "not_dpl.hdf5", dummy_data, overwrite=True)
    with pytest.raises(ValueError, match="The object should be of type Dipole."):
        read_dipole(tmp_path / "not_dpl.hdf5")
    # force the scale_applied to be identical across dpls to allow averaging.
    dipole.scale_applied = dipole_read.scale_applied
    # average two identical dipole objects
    dipole_avg = average_dipoles([dipole, dipole_read])
    for dpl_key in dipole_avg.data.keys():
        assert_allclose(
            dipole_read.data[dpl_key], dipole_avg.data[dpl_key], rtol=0, atol=0.000051
        )

    with pytest.raises(
        ValueError, match="Dipole at index 0 was already an average of 2 trials"
    ):
        dipole_avg = average_dipoles([dipole_avg, dipole_read])

    # average an n_of_1 dipole list
    single_dpl_avg = average_dipoles([dipole])
    for dpl_key in single_dpl_avg.data.keys():
        assert_allclose(
            dipole_read.data[dpl_key],
            single_dpl_avg.data[dpl_key],
            rtol=0,
            atol=0.000051,
        )

    # average dipole list with one dipole object and a zero dipole object
    n_times = len(dipole_read.data["agg"])
    dpl_null = Dipole(
        np.zeros(
            n_times,
        ),
        np.zeros((n_times, 3)),
    )
    dpl_1 = [dipole, dpl_null]
    dpl_avg = average_dipoles(dpl_1)
    for dpl_key in dpl_avg.data.keys():
        assert_allclose(dpl_1[0].data[dpl_key] / 2.0, dpl_avg.data[dpl_key])

    # Test experimental dipole
    dipole_exp = Dipole(times, data[:, 1])
    dipole_exp.write(dpl_out_fname)
    dipole_exp_read = read_dipole(dpl_out_fname)
    assert_allclose(dipole_exp.data["agg"], dipole_exp_read.data["agg"], rtol=1e-2)
    dipole_exp_avg = average_dipoles([dipole_exp, dipole_exp])
    assert_allclose(dipole_exp.data["agg"], dipole_exp_avg.data["agg"])

    # XXX all below to be deprecated in 0.3
    dpls_raw, net = run_hnn_core_fixture(
        backend="joblib",
        n_jobs=1,
        reduced=True,
        record_isec="soma",
        record_vsec="soma",
        record_ca="soma",
    )
    # test deprecation of postproc
    with pytest.warns(DeprecationWarning, match="The postproc-argument is deprecated"):
        dpls, _ = run_hnn_core_fixture(
            backend="joblib",
            n_jobs=1,
            reduced=True,
            record_isec="soma",
            record_vsec="soma",
            record_ca="soma",
            postproc=True,
        )
    with pytest.raises(AssertionError):
        assert_allclose(dpls[0].data["agg"], dpls_raw[0].data["agg"])

    dpls_raw[0]._post_proc(
        net._params["dipole_smooth_win"], net._params["dipole_scalefctr"]
    )
    assert_allclose(dpls_raw[0].data["agg"], dpls[0].data["agg"])

    plt.close("all")


def test_dipole_simulation():
    """Test data produced from simulate_dipole() call."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)
    params.update(
        {"dipole_smooth_win": 5, "t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20}
    )
    net = jones_2009_model(params, add_drives_from_params=True, mesh_shape=(3, 3))
    with pytest.raises(ValueError, match="Invalid number of simulations: 0"):
        simulate_dipole(net, tstop=25.0, n_trials=0)
    with pytest.raises(ValueError, match="Invalid value for the"):
        simulate_dipole(net, tstop=25.0, n_trials=1, record_vsec="abc")
    with pytest.raises(ValueError, match="Invalid value for the"):
        simulate_dipole(
            net, tstop=25.0, n_trials=1, record_vsec=False, record_isec="abc"
        )
    with pytest.raises(ValueError, match="Invalid value for the"):
        simulate_dipole(
            net,
            tstop=25.0,
            n_trials=1,
            record_vsec=False,
            record_isec=False,
            record_ca="abc",
        )

    # test Network.copy() returns 'bare' network after simulating
    dpl = simulate_dipole(net, tstop=25.0, n_trials=1)[0]
    assert net._dt == 0.025
    assert net._tstop == 25.0
    net_copy = net.copy()
    assert len(net_copy.external_drives["evprox1"]["events"]) == 0

    # test that Dipole.copy() returns the expected exact copy
    assert_allclose(dpl.data["agg"], dpl.copy().data["agg"])

    with pytest.warns(UserWarning, match="No connections"):
        net = Network(params)
        # warning triggered on simulate_dipole()
        simulate_dipole(net, tstop=0.1, n_trials=1)

        # Smoke test for raster plot with no spikes
        net.cell_response.plot_spikes_raster()
    plt.close("all")


@requires_mpi4py
@requires_psutil
@pytest.mark.uses_mpi
def test_cell_response_backends(run_hnn_core_fixture):
    """Test cell_response outputs across backends."""

    # reduced simulation has n_trials=2
    trial_idx, n_trials, gid = 0, 2, 7
    _, joblib_net = run_hnn_core_fixture(
        backend="joblib",
        n_jobs=1,
        reduced=True,
        record_vsec="all",
        record_isec="soma",
        record_ca="all",
    )
    _, mpi_net = run_hnn_core_fixture(
        backend="mpi",
        n_procs=2,
        reduced=True,
        record_vsec="all",
        record_isec="soma",
        record_ca="all",
    )

    n_times = len(joblib_net.cell_response.times)

    assert len(joblib_net.cell_response.vsec) == n_trials
    assert len(joblib_net.cell_response.isec) == n_trials
    assert len(joblib_net.cell_response.ca) == n_trials
    assert len(joblib_net.cell_response.vsec[trial_idx][gid]) == 8  # num sec
    assert len(joblib_net.cell_response.isec[trial_idx][gid]) == 1
    assert len(joblib_net.cell_response.vsec[trial_idx][gid]["apical_1"]) == n_times
    assert (
        len(joblib_net.cell_response.isec[trial_idx][gid]["soma"]["soma_gabaa"])
        == n_times
    )
    assert len(mpi_net.cell_response.vsec) == n_trials
    assert len(mpi_net.cell_response.isec) == n_trials
    assert len(mpi_net.cell_response.vsec[trial_idx][gid]) == 8  # num sec
    assert len(mpi_net.cell_response.isec[trial_idx][gid]) == 1
    assert len(mpi_net.cell_response.vsec[trial_idx][gid]["apical_1"]) == n_times
    assert (
        len(mpi_net.cell_response.isec[trial_idx][gid]["soma"]["soma_gabaa"]) == n_times
    )
    assert mpi_net.cell_response.vsec == joblib_net.cell_response.vsec
    assert mpi_net.cell_response.isec == joblib_net.cell_response.isec

    # test if calcium concentration is stored correctly (only L5 pyramidal)
    gid = joblib_net.gid_ranges["L5_pyramidal"][0]
    assert len(joblib_net.cell_response.ca[trial_idx][gid]) == 9
    assert len(joblib_net.cell_response.ca[trial_idx][gid]["soma"]) == n_times
    assert len(mpi_net.cell_response.ca[trial_idx][gid]) == 9
    assert len(mpi_net.cell_response.ca[trial_idx][gid]["soma"]) == n_times
    assert len(mpi_net.cell_response.ca[trial_idx][gid]["apical_1"]) == n_times
    # Test if spike time falls within depolarization window above v_thresh
    v_thresh = 0.0
    times = np.array(joblib_net.cell_response.times)
    spike_times = np.array(joblib_net.cell_response.spike_times[trial_idx])
    spike_gids = np.array(joblib_net.cell_response.spike_gids[trial_idx])
    vsoma = np.array(joblib_net.cell_response.vsec[trial_idx][gid]["soma"])

    v_mask = vsoma > v_thresh
    assert np.all(
        [
            spike_times[spike_gids == gid] > times[v_mask][0],
            spike_times[spike_gids == gid] < times[v_mask][-1],
        ]
    )

    # test that event times before and after simulation are the same
    for drive_name, drive in joblib_net.external_drives.items():
        gid_ran = joblib_net.gid_ranges[drive_name]
        for idx_drive, event_times in enumerate(drive["events"][trial_idx]):
            net_ets = [
                spike_times[i]
                for i, g in enumerate(spike_gids)
                if g == gid_ran[idx_drive]
            ]
            assert_allclose(np.array(event_times), np.array(net_ets))


def test_rmse():
    """Test to check RMSE calculation"""
    data_url = (
        "https://raw.githubusercontent.com/jonescompneurolab/hnn/"
        "master/data/MEG_detection_data/yes_trial_S1_ERP_all_avg.txt"
    )
    if not op.exists("yes_trial_S1_ERP_all_avg.txt"):
        urlretrieve(data_url, "yes_trial_S1_ERP_all_avg.txt")
    extdata = np.loadtxt("yes_trial_S1_ERP_all_avg.txt")

    exp_dpl = Dipole(
        times=extdata[:, 0], data=np.c_[extdata[:, 1], extdata[:, 1], extdata[:, 1]]
    )

    hnn_core_root = op.join(op.dirname(hnn_core.__file__))
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)

    expected_rmse = 0.1
    test_dpl = Dipole(
        times=extdata[:, 0],
        data=np.c_[
            extdata[:, 1] + expected_rmse,
            extdata[:, 1] + expected_rmse,
            extdata[:, 1] + expected_rmse,
        ],
    )
    avg_rmse = _rmse(test_dpl, exp_dpl, tstop=params["tstop"])

    assert_allclose(avg_rmse, expected_rmse)
