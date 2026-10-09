from pathlib import Path
from os import environ
import io
import itertools
from contextlib import redirect_stdout
from threading import Thread, Event
from time import sleep
from urllib.request import urlretrieve

import numpy as np
from numpy import loadtxt
from numpy.testing import assert_array_equal, assert_allclose, assert_raises

import pytest

import hnn_core
from hnn_core import (
    MPIBackend,
    neymotin_2020_model,
    read_params,
    read_spikes,
)
from hnn_core.dipole import simulate_dipole
from hnn_core.parallel_backends import (
    requires_mpi4py,
    requires_psutil,
    _determine_cores_hwthreading,
    _get_mpi_env,
    _get_pip_openmpi_lib,
)
from hnn_core.network_builder import NetworkBuilder


def _has_pip_openmpi():
    """Check whether the PyPI 'openmpi' package is installed."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("openmpi")
    except PackageNotFoundError:
        return False
    return True


requires_pip_openmpi = pytest.mark.skipif(
    not _has_pip_openmpi(), reason="requires the PyPI 'openmpi' package"
)


def _terminate_mpibackend(event, backend):
    # wait for run_subprocess to start MPI proc and put handle on queue
    proc = backend.proc_queue.get()
    # put the proc back in the queue. used by backend.terminate()
    backend.proc_queue.put(proc)

    # give the process a little time to startup
    sleep(0.1)

    # run terminate until it is successful
    while not event.is_set():
        backend.terminate()
        sleep(0.01)


def test_gid_assignment():
    """Test that gids are assigned without overlap across ranks"""

    net = neymotin_2020_model(add_drives_from_params=False)
    weights_ampa = {"L2_basket": 1.0, "L2_pyramidal": 2.0, "L5_pyramidal": 3.0}
    syn_delays = {"L2_basket": 0.1, "L2_pyramidal": 0.2, "L5_pyramidal": 0.3}

    net.add_bursty_drive(
        "bursty_dist",
        location="distal",
        burst_rate=10,
        weights_ampa=weights_ampa,
        synaptic_delays=syn_delays,
        cell_specific=False,
        n_drive_cells=5,
    )
    net.add_evoked_drive(
        "evoked_prox",
        mu=1.0,
        sigma=1.0,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="proximal",
        synaptic_delays=syn_delays,
        cell_specific=True,
        n_drive_cells="n_cells",
    )
    net._instantiate_drives(tstop=20, n_trials=2)

    all_gids = list()
    for type_range in net.gid_ranges.values():
        all_gids.extend(list(type_range))
    all_gids.sort()

    n_hosts = 3
    all_gids_instantiated = list()
    for rank in range(n_hosts):
        net_builder = NetworkBuilder(net)
        net_builder._gid_list = list()
        net_builder._gid_assign(rank=rank, n_hosts=n_hosts)
        all_gids_instantiated.extend(net_builder._gid_list)
    all_gids_instantiated.sort()
    assert all_gids_instantiated == sorted(set(all_gids_instantiated))
    assert all_gids == all_gids_instantiated


# The purpose of this incremental mark is to avoid running the full length
# simulation when there are failures in previous (faster) tests. When a test
# in the sequence fails, all subsequent tests will be marked "xfailed" rather
# than skipped.
@pytest.mark.incremental
@pytest.mark.uses_mpi
class TestParallelBackends:
    dpls_reduced_mpi = None
    dpls_reduced_default = None
    dpls_reduced_joblib = None

    def test_run_default(self, run_hnn_core_fixture):
        """Test consistency between default backend simulation and master"""
        global dpls_reduced_default
        dpls_reduced_default, _ = run_hnn_core_fixture(None, reduced=True)
        # test consistency across all parallel backends for multiple trials
        assert_raises(
            AssertionError,
            assert_array_equal,
            dpls_reduced_default[0].data["agg"],
            dpls_reduced_default[1].data["agg"],
        )

    def test_run_joblibbackend(self, run_hnn_core_fixture):
        """Test consistency between joblib backend simulation with master"""
        global dpls_reduced_default, dpls_reduced_joblib

        dpls_reduced_joblib, _ = run_hnn_core_fixture(
            backend="joblib", n_jobs=2, reduced=True
        )

        for trial_idx in range(len(dpls_reduced_default)):
            assert_array_equal(
                dpls_reduced_default[trial_idx].data["agg"],
                dpls_reduced_joblib[trial_idx].data["agg"],
            )

    @requires_mpi4py
    @requires_psutil
    @pytest.mark.parametrize("sensible_default", [False, True])
    def test_detect_cores(self, sensible_default):
        """Test that multiple cores can be detected"""
        [detected_cores_nohw, detected_hwthreading] = _determine_cores_hwthreading(
            use_hwthreading_if_found=False, sensible_default_cores=sensible_default
        )
        assert detected_cores_nohw > 1
        assert isinstance(detected_hwthreading, bool)

        [detected_cores_yeshw, detected_hwthreading] = _determine_cores_hwthreading(
            use_hwthreading_if_found=True, sensible_default_cores=sensible_default
        )
        assert detected_cores_yeshw > 1
        assert isinstance(detected_hwthreading, bool)

        assert detected_cores_yeshw >= detected_cores_nohw

    @requires_mpi4py
    @requires_psutil
    def test_mpi_nprocs(self):
        """Test that MPIBackend can use more than 1 processor"""
        # if only 1 processor is available, then MPIBackend tests will not
        # be valid
        with MPIBackend() as backend:
            assert backend.n_procs > 1

    @requires_mpi4py
    @requires_psutil
    def test_run_mpibackend(self, run_hnn_core_fixture):
        """Test running a MPIBackend on reduced model"""
        global dpls_reduced_default, dpls_reduced_mpi
        dpls_reduced_mpi, _ = run_hnn_core_fixture(backend="mpi", reduced=True)
        for trial_idx in range(len(dpls_reduced_default)):
            # account for rounding error incured during MPI parallelization
            assert_allclose(
                dpls_reduced_default[trial_idx].data["agg"],
                dpls_reduced_mpi[trial_idx].data["agg"],
                rtol=0,
                atol=1e-14,
            )

    @requires_mpi4py
    @requires_psutil
    def test_terminate_mpibackend(self, run_hnn_core_fixture):
        """Test terminating MPIBackend from thread"""
        hnn_core_root = Path(hnn_core.__file__).parent
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
        params.update(
            {"t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20, "N_trials": 2}
        )
        net = neymotin_2020_model(
            params, add_drives_from_params=True, mesh_shape=(3, 3)
        )

        with MPIBackend() as backend:
            event = Event()
            # start background thread that will kill all MPIBackends
            # until event.set()
            kill_t = Thread(target=_terminate_mpibackend, args=(event, backend))
            # make thread a daemon in case we throw an exception
            # and don't run event.set() so that py.test will
            # not hang before exiting
            kill_t.daemon = True
            kill_t.start()

            with pytest.warns(UserWarning) as record:
                with pytest.raises(
                    RuntimeError, match="MPI simulation failed. Return code: 1"
                ):
                    simulate_dipole(net, tstop=40)

            event.set()
        expected_string = "Child process failed unexpectedly"
        assert expected_string in record[0].message.args[0]

    @requires_mpi4py
    @requires_psutil
    @pytest.mark.parametrize("use_hwthreading_if_found", [True, False])
    def test_run_mpibackend_oversubscribed(self, use_hwthreading_if_found):
        """Test running MPIBackend with oversubscribed number of procs"""
        hnn_core_root = Path(hnn_core.__file__).parent
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
        params.update(
            {"t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20, "N_trials": 2}
        )
        net = neymotin_2020_model(
            params, add_drives_from_params=True, mesh_shape=(3, 3)
        )

        # Fail state: try running with more procs than cells in the network
        # (will probably oversubscribe too)
        too_many_procs = net._n_cells + 1

        with pytest.raises(
            ValueError, match=("More MPI processes were assigned than there are cells")
        ):
            with MPIBackend(n_procs=too_many_procs) as backend:
                simulate_dipole(net, tstop=40)

        # Force oversubscription and make sure there are always enough cells in
        # the network
        [detected_cores, detected_hwthreading] = _determine_cores_hwthreading(
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=False,
        )

        oversubscribed_procs = detected_cores + 1
        n_grid_1d = int(np.ceil(np.sqrt(oversubscribed_procs)))
        params.update(
            {"t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20, "N_trials": 2}
        )
        net = neymotin_2020_model(
            params, add_drives_from_params=True, mesh_shape=(n_grid_1d, n_grid_1d)
        )

        # Case 1 (default): Check that oversubscription turns on if needed, and
        # provides a warning
        override_oversubscribe_option = None
        with pytest.warns(
            UserWarning,
            match=(
                "Number of requested MPI processes exceeds "
                "available cores. Enabling MPI "
                "oversubscription automatically."
            ),
        ):
            with MPIBackend(
                n_procs=oversubscribed_procs,
                use_hwthreading_if_found=use_hwthreading_if_found,
                override_oversubscribe_option=override_oversubscribe_option,
            ) as backend:
                assert backend.n_procs == oversubscribed_procs
                assert "--oversubscribe" in " ".join(backend.mpi_cmd)
                simulate_dipole(net, tstop=40)

        # Case 2: Check that '--oversubscribe' option is passed if
        # oversubscription is forced on. No simulation is run in this case
        # since MacOS CI runners randomly (but not always) fail to run in the
        # following case, even though the same simulation succeeds in Case 1
        # above. This is probably due to MPI instability with the MacOS CI
        # runners (see #992 for an example).
        override_oversubscribe_option = True
        with MPIBackend(
            n_procs=oversubscribed_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            override_oversubscribe_option=override_oversubscribe_option,
        ) as backend:
            assert "--oversubscribe" in " ".join(backend.mpi_cmd)

        # Case 3: Check that the simulation fails if oversubscribe is forced
        # off
        override_oversubscribe_option = False
        with MPIBackend(
            n_procs=oversubscribed_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            override_oversubscribe_option=override_oversubscribe_option,
        ) as backend:
            assert "--oversubscribe" not in " ".join(backend.mpi_cmd)
            with pytest.raises(
                RuntimeError, match="MPI simulation failed. Return code: 1"
            ):
                simulate_dipole(net, tstop=40)

    @requires_mpi4py
    @requires_psutil
    @pytest.mark.parametrize(
        "use_hwthreading_if_found,sensible_default_cores,override_oversubscribe_option",
        [
            x
            for x in itertools.product(
                [True, False], [True, False], [None, True, False]
            )
        ],
    )
    def test_run_mpibackend_hwthreading(
        self,
        use_hwthreading_if_found,
        sensible_default_cores,
        override_oversubscribe_option,
    ):
        """Test running MPIBackend with oversubscribed number of procs"""
        hnn_core_root = Path(hnn_core.__file__).parent
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
        params.update(
            {"t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20, "N_trials": 2}
        )
        net = neymotin_2020_model(
            params, add_drives_from_params=True, mesh_shape=(3, 3)
        )

        n_procs = 2
        # Test that the network runs at all
        with MPIBackend(n_procs=n_procs) as backend:
            simulate_dipole(net, tstop=40)

        [_, detected_hwthreading] = _determine_cores_hwthreading(
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
        )

        # Case 1 (default): Check that hwthreading turns on if needed
        override_hwthreading_option = None

        # Possibly needed to prevent MPIBackend failures to exit processes
        del net
        net = neymotin_2020_model(
            params, add_drives_from_params=True, mesh_shape=(3, 3)
        )
        with MPIBackend(
            n_procs=n_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
            override_oversubscribe_option=override_oversubscribe_option,
            override_hwthreading_option=override_hwthreading_option,
        ) as backend:
            if detected_hwthreading:
                assert "--use-hwthread-cpus" in " ".join(backend.mpi_cmd)
            simulate_dipole(net, tstop=40)

        # Case 2: Check that hwthreading turns on if forced. Note that the
        # simulation should NOT be run in this case, since the underlying
        # hardware will NOT necessarily have hardware-threading available.
        override_hwthreading_option = True
        with MPIBackend(
            n_procs=n_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
            override_oversubscribe_option=override_oversubscribe_option,
            override_hwthreading_option=override_hwthreading_option,
        ) as backend:
            assert "--use-hwthread-cpus" in " ".join(backend.mpi_cmd)

        # Case 3: Check that hwthreading turns off if forced off.
        override_hwthreading_option = False

        # Possibly needed to prevent MPIBackend failures to exit processes
        del net
        net = neymotin_2020_model(
            params, add_drives_from_params=True, mesh_shape=(3, 3)
        )
        with MPIBackend(
            n_procs=n_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
            override_oversubscribe_option=override_oversubscribe_option,
            override_hwthreading_option=override_hwthreading_option,
        ) as backend:
            assert "--use-hwthread-cpus" not in " ".join(backend.mpi_cmd)
            simulate_dipole(net, tstop=40)

    @pytest.mark.parametrize("backend", ["mpi", "joblib"])
    def test_compare_hnn_core(self, run_hnn_core_fixture, backend, n_jobs=1):
        """Test hnn-core does not break."""
        # small snippet of data on data branch for now. To be deleted
        # later. Data branch should have only commit so it does not
        # pollute the history.
        data_url = (
            "https://raw.githubusercontent.com/jonescompneurolab/"
            "hnn-core/test_data/dpl.txt"
        )
        if not Path("dpl.txt").exists():
            urlretrieve(data_url, "dpl.txt")
        dpl_master = loadtxt("dpl.txt")

        dpls, net = run_hnn_core_fixture(backend=backend)
        dpl = dpls[0].smooth(30).scale(3000)

        # write the dipole to a file and compare
        fname = "./dpl2.txt"
        dpl.write(fname)

        dpl_pr = loadtxt(fname)
        assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
        assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5

        # Test spike type counts
        spike_type_counts = {}
        for spike_gid in net.cell_response.spike_gids[0]:
            if net.gid_to_type(spike_gid) not in spike_type_counts:
                spike_type_counts[net.gid_to_type(spike_gid)] = 1
            else:
                spike_type_counts[net.gid_to_type(spike_gid)] += 1
        assert "common" not in spike_type_counts
        assert "exgauss" not in spike_type_counts
        assert "extpois" not in spike_type_counts
        assert spike_type_counts == {
            "evprox1": 270,
            "L2_basket": 55,
            "L2_pyramidal": 114,
            "L5_pyramidal": 396,
            "L5_basket": 86,
            "evdist1": 270,
            "evprox2": 270,
        }


# there are no dependencies if this unit tests fails; no need to be in
# class marked incremental
@requires_mpi4py
@requires_psutil
@pytest.mark.uses_mpi
def test_mpi_failure(run_hnn_core_fixture):
    """Test that an MPI failure is handled and messages are printed"""
    # this MPI parameter will cause a MPI job to fail
    environ["OMPI_MCA_btl"] = "self"

    with pytest.warns(UserWarning) as record:
        with io.StringIO() as buf, redirect_stdout(buf):
            with pytest.raises(RuntimeError, match="MPI simulation failed"):
                run_hnn_core_fixture(backend="mpi", reduced=True, postproc=False)
            stdout = buf.getvalue()

    assert "MPI processes are unable to reach each other" in stdout

    expected_string = "Child process failed unexpectedly"
    assert len(record) == 1
    assert record[0].message.args[0] == expected_string

    del environ["OMPI_MCA_btl"]


@pytest.mark.uses_mpi
@pytest.mark.duecker
@pytest.mark.parametrize("backend", ["mpi", "joblib"])
def test_compare_duecker_model_output(backend):
    """Test that the Duecker model dipole output does not change"""

    from script_duecker_simulate_save import rerun_and_save_duecker_model

    # Run the Duecker model, save dipole and spiking output to files, then reload data
    # ----------------------------------------------------------------------------------
    # This creates two new files, `dipole_duecker_output_new.txt` and
    # `spikes_duecker_output_new.txt` in the local directory..
    rerun_and_save_duecker_model(suffix="new", backend=backend)

    dpl_new_reloaded = loadtxt("dipole_duecker_output_new.txt")
    cell_response_new_reloaded = read_spikes("spikes_duecker_output_new.txt")

    # Load the old "ground-truth" dipole and spike data
    # ----------------------------------------------------------------------------------
    # The below single-trial dipole output data file was generated by running
    # ```
    # python ./hnn_core/tests/script_duecker_simulate_save.py
    # ```
    # as of
    # https://github.com/katduecker/hnn-core/commit/5298d5c0c3e9e72f69404b9327dd1db4aa900ffd
    # Note: If you need to regenerate the "old" data, then where you run the script from
    # will determine where the output files are saved, so make sure to run the script
    # from the "top-level" repository directory (i.e. the one that contains the
    # `setup.py` file).
    dpl_old = loadtxt("dipole_duecker_output_old.txt")
    cell_response_old = read_spikes("spikes_duecker_output_old.txt")

    # Compare our dipole data EXACTLY (single-float precision)
    # ----------------------------------------------------------------------------------
    assert_array_equal(dpl_new_reloaded[:, 2], dpl_old[:, 2])  # L2
    assert_array_equal(dpl_new_reloaded[:, 3], dpl_old[:, 3])  # L5

    # Compare our spike times EXACTLY (single-float precision)
    # ----------------------------------------------------------------------------------
    assert_array_equal(
        cell_response_new_reloaded.spike_times,
        cell_response_old.spike_times,
    )


# --------------------------------------------------------------------------------------
# Tests for locating and loading the MPI library ('MPI_LIB_NRN_PATH')
# --------------------------------------------------------------------------------------
class _FakePackageFile:
    """Minimal stand-in for an importlib.metadata.PackagePath."""

    def __init__(self, location):
        self.name = Path(location).name
        self._location = location

    def locate(self):
        return self._location


class _FakeDistribution:
    """Minimal stand-in for an importlib.metadata.Distribution."""

    def __init__(self, files):
        self.files = files


def _fake_openmpi_package(monkeypatch, file_paths):
    """Make the PyPI 'openmpi' package appear to contain 'file_paths'."""
    import importlib.metadata

    files = [_FakePackageFile(path) for path in file_paths]
    # _get_pip_openmpi_lib imports 'distribution' at call time, so we patch the source
    # module rather than hnn_core.parallel_backends
    monkeypatch.setattr(
        importlib.metadata, "distribution", lambda name: _FakeDistribution(files)
    )


def _fake_pip_openmpi_lib(monkeypatch, mpi_lib):
    """Make _get_pip_openmpi_lib return 'mpi_lib' without searching anything."""
    from hnn_core import parallel_backends

    monkeypatch.setattr(parallel_backends, "_get_pip_openmpi_lib", lambda: mpi_lib)


class TestGetPipOpenmpiLib:
    """Tests for locating libmpi inside the PyPI 'openmpi' package"""

    # Files that look similar to the real library but must NOT be matched. Notably,
    # 'libmpi.so' and 'libmpi.dylib' are linker scripts that NEURON cannot load.
    DECOY_FILES = [
        "libmpi.so",
        "libmpi.dylib",
        "libmpi_mpifh.so.40",
        "libmpi.so.40.1.0",
    ]

    @pytest.mark.parametrize(
        "lib_name",
        ["libmpi.so.40", "libmpi.40.dylib", "libmpi.so.12", "libmpi.12.dylib"],
    )
    def test_lib_found(self, monkeypatch, tmp_path, lib_name):
        """Test that the real libmpi file is found among the decoys"""
        lib_dir = tmp_path / "lib"
        decoys = [lib_dir / name for name in self.DECOY_FILES]
        # Use a non-normalized path to check that the result gets normalized
        real_lib = lib_dir / ".." / "lib" / lib_name
        _fake_openmpi_package(monkeypatch, decoys + [real_lib])

        assert _get_pip_openmpi_lib() == str(lib_dir / lib_name)

    def test_only_decoys(self, monkeypatch, tmp_path):
        """Test that None is returned if no file matches"""
        decoys = [tmp_path / name for name in self.DECOY_FILES]
        _fake_openmpi_package(monkeypatch, decoys)

        assert _get_pip_openmpi_lib() is None

    def test_empty_package(self, monkeypatch):
        """Test that None is returned if the package lists no files"""
        _fake_openmpi_package(monkeypatch, [])

        assert _get_pip_openmpi_lib() is None

    def test_not_installed(self, monkeypatch, capsys):
        """Test that None is returned and a message printed if not installed"""
        import importlib.metadata

        def _not_installed(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "distribution", _not_installed)

        assert _get_pip_openmpi_lib() is None
        assert "PyPI 'openmpi' package not found" in capsys.readouterr().out

    @requires_pip_openmpi
    def test_real_package(self):
        """Test that libmpi is found in an actually-installed 'openmpi' package"""
        mpi_lib = _get_pip_openmpi_lib()

        assert mpi_lib is not None
        assert Path(mpi_lib).is_absolute()
        assert Path(mpi_lib).is_file()


# Fake library paths. Each one names where the MPI library "came from", so a
# failing test makes it obvious which source _get_mpi_env chose.
PREEXISTING_LIB = "/preexisting/libmpi.so.40"  # already in the caller's env
PIP_LIB = "/pip/libmpi.so.40"  # found inside the PyPI 'openmpi' package
USER_LIB = "/user/libmpi.so.40"  # passed explicitly via `mpi_lib_path`


def _lib_path_case(
    test_id,
    *,
    preexisting_env_value,
    pip_lib_found,
    autoload,
    user_lib_path,
    expected_env_value,
):
    """Build one test case for test_get_mpi_env_lib_path.

    Parameters
    ----------
    test_id : str
        Name shown by pytest for this case.
    preexisting_env_value : str | None
        Value of MPI_LIB_NRN_PATH in the caller's environment before the call,
        or None if it is unset.
    pip_lib_found : str | None
        Path that the (faked) PyPI 'openmpi' library search returns, or None
        if the search finds nothing.
    autoload : bool
        Value passed as `autoload_mpi_library`.
    user_lib_path : str | Path | None
        Value passed as `mpi_lib_path`.
    expected_env_value : str | None
        Expected value of MPI_LIB_NRN_PATH in the returned env, or None if it
        should be absent.
    """
    return pytest.param(
        preexisting_env_value,
        pip_lib_found,
        autoload,
        user_lib_path,
        expected_env_value,
        id=test_id,
    )


@pytest.mark.parametrize(
    "preexisting_env_value, pip_lib_found, autoload, user_lib_path, expected_env_value",
    [
        # ---------------------------------------------------------------------
        # Autoloading enabled, no user path: the PyPI 'openmpi' library is
        # used, overwriting any pre-existing value
        # ---------------------------------------------------------------------
        _lib_path_case(
            "autoload",
            preexisting_env_value=None,
            pip_lib_found=PIP_LIB,
            autoload=True,
            user_lib_path=None,
            expected_env_value=PIP_LIB,
        ),
        _lib_path_case(
            "autoload-overwrites",
            preexisting_env_value=PREEXISTING_LIB,
            pip_lib_found=PIP_LIB,
            autoload=True,
            user_lib_path=None,
            expected_env_value=PIP_LIB,
        ),
        # ---------------------------------------------------------------------
        # Autoloading enabled, but no PyPI 'openmpi' library found: the
        # environment is left alone
        # ---------------------------------------------------------------------
        _lib_path_case(
            "autoload-not-found",
            preexisting_env_value=None,
            pip_lib_found=None,
            autoload=True,
            user_lib_path=None,
            expected_env_value=None,
        ),
        _lib_path_case(
            "autoload-not-found-keeps-preexisting",
            preexisting_env_value=PREEXISTING_LIB,
            pip_lib_found=None,
            autoload=True,
            user_lib_path=None,
            expected_env_value=PREEXISTING_LIB,
        ),
        # ---------------------------------------------------------------------
        # Autoloading disabled, no user path: the environment is left alone,
        # even though a PyPI 'openmpi' library is available
        # ---------------------------------------------------------------------
        _lib_path_case(
            "no-autoload",
            preexisting_env_value=None,
            pip_lib_found=PIP_LIB,
            autoload=False,
            user_lib_path=None,
            expected_env_value=None,
        ),
        _lib_path_case(
            "no-autoload-keeps-preexisting",
            preexisting_env_value=PREEXISTING_LIB,
            pip_lib_found=PIP_LIB,
            autoload=False,
            user_lib_path=None,
            expected_env_value=PREEXISTING_LIB,
        ),
        # ---------------------------------------------------------------------
        # User-supplied path (str or Path): always wins, regardless of
        # autoloading or any pre-existing value
        # ---------------------------------------------------------------------
        _lib_path_case(
            "user-str",
            preexisting_env_value=None,
            pip_lib_found=PIP_LIB,
            autoload=True,
            user_lib_path=USER_LIB,
            expected_env_value=USER_LIB,
        ),
        _lib_path_case(
            "user-path",
            preexisting_env_value=None,
            pip_lib_found=PIP_LIB,
            autoload=True,
            user_lib_path=Path(USER_LIB),
            expected_env_value=USER_LIB,
        ),
        _lib_path_case(
            "user-with-no-autoload",
            preexisting_env_value=None,
            pip_lib_found=PIP_LIB,
            autoload=False,
            user_lib_path=USER_LIB,
            expected_env_value=USER_LIB,
        ),
        _lib_path_case(
            "user-overwrites-preexisting",
            preexisting_env_value=PREEXISTING_LIB,
            pip_lib_found=PIP_LIB,
            autoload=True,
            user_lib_path=USER_LIB,
            expected_env_value=USER_LIB,
        ),
    ],
)
def test_get_mpi_env_lib_path(
    monkeypatch,
    preexisting_env_value,
    pip_lib_found,
    autoload,
    user_lib_path,
    expected_env_value,
):
    """Test how _get_mpi_env sets MPI_LIB_NRN_PATH"""
    # Set up the caller's environment
    if preexisting_env_value is None:
        monkeypatch.delenv("MPI_LIB_NRN_PATH", raising=False)
    else:
        monkeypatch.setenv("MPI_LIB_NRN_PATH", preexisting_env_value)

    # Control what the PyPI 'openmpi' library search returns
    _fake_pip_openmpi_lib(monkeypatch, pip_lib_found)

    env = _get_mpi_env(autoload_mpi_library=autoload, mpi_lib_path=user_lib_path)

    # The returned env has the expected library path
    assert env.get("MPI_LIB_NRN_PATH") == expected_env_value
    # The caller's actual environment must never be modified
    assert environ.get("MPI_LIB_NRN_PATH") == preexisting_env_value


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(autoload_mpi_library=False), id="no-autoload"),
        pytest.param(dict(mpi_lib_path=USER_LIB), id="user-path"),
    ],
)
def test_get_mpi_env_skips_openmpi_search(monkeypatch, capsys, kwargs):
    """Test that the 'openmpi' package is only searched when autoloading"""
    from hnn_core import parallel_backends

    def _fail():
        raise AssertionError("'openmpi' package should not be searched")

    monkeypatch.setattr(parallel_backends, "_get_pip_openmpi_lib", _fail)

    _get_mpi_env(**kwargs)
    assert capsys.readouterr().out == ""


@requires_mpi4py
@requires_psutil
@pytest.mark.parametrize("autoload", [True, False])
def test_mpibackend_autoload_without_lib_path(autoload):
    """Test that MPIBackend stores 'autoload_mpi_library' as given"""
    backend = MPIBackend(autoload_mpi_library=autoload)

    assert backend.autoload_mpi_library is autoload
    assert backend.mpi_lib_path is None


@requires_mpi4py
@requires_psutil
@pytest.mark.parametrize("path_type", ["path", "str", "relative-str"])
def test_mpibackend_lib_path(monkeypatch, tmp_path, path_type):
    """Test that 'mpi_lib_path' is made absolute and disables autoloading"""
    lib_file = tmp_path / "libmpi.so.40"
    lib_file.touch()
    monkeypatch.chdir(tmp_path)
    mpi_lib_path = {
        "path": lib_file,
        "str": str(lib_file),
        "relative-str": "libmpi.so.40",
    }[path_type]

    backend = MPIBackend(autoload_mpi_library=True, mpi_lib_path=mpi_lib_path)

    assert backend.autoload_mpi_library is False
    assert backend.mpi_lib_path == str(lib_file)


@requires_mpi4py
@requires_psutil
@pytest.mark.parametrize("bad_path", ["nonexistent.so", "."], ids=["missing", "dir"])
def test_mpibackend_lib_path_not_a_file(tmp_path, bad_path):
    """Test that 'mpi_lib_path' must be an existing file"""
    with pytest.raises(FileNotFoundError, match="'mpi_lib_path' not found"):
        MPIBackend(mpi_lib_path=tmp_path / bad_path)


@requires_mpi4py
@requires_psutil
def test_mpibackend_passes_mpi_lib_args(monkeypatch, tmp_path):
    """Test that MPIBackend.simulate passes its MPI library args to _get_mpi_env"""
    from hnn_core import parallel_backends

    class _StopSimulation(Exception):
        pass

    received_kwargs = []

    def _fake_get_mpi_env(**kwargs):
        # Record the arguments, then abort before any MPI process is started
        received_kwargs.append(kwargs)
        raise _StopSimulation

    monkeypatch.setattr(parallel_backends, "_get_mpi_env", _fake_get_mpi_env)

    hnn_core_root = Path(hnn_core.__file__).parent
    params = read_params(hnn_core_root / "param" / "default.json")
    net = neymotin_2020_model(params, add_drives_from_params=False, mesh_shape=(3, 3))
    lib_file = tmp_path / "libmpi.so.40"
    lib_file.touch()
    backend = MPIBackend(n_procs=2, mpi_lib_path=lib_file)

    with pytest.raises(_StopSimulation):
        backend.simulate(net, tstop=1, dt=0.025, n_trials=1)
    assert received_kwargs == [
        dict(autoload_mpi_library=False, mpi_lib_path=str(lib_file))
    ]
