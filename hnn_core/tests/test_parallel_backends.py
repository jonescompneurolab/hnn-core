from pathlib import Path
from os import environ
import io
import itertools
import re
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, distribution
from threading import Thread, Event
from time import sleep
from urllib.request import urlretrieve

import numpy as np
from numpy import loadtxt
from numpy.testing import assert_array_equal, assert_allclose, assert_raises

import pytest

from hnn_core import parallel_backends
from hnn_core import (
    MPIBackend,
    neymotin_2020_model,
    read_spikes,
)
from hnn_core.dipole import simulate_dipole
from hnn_core.parallel_backends import (
    requires_mpi4py,
    requires_psutil,
    requires_openmpi_package,
    _determine_cores_hwthreading,
    _get_mpi_env,
    _get_pip_openmpi_lib,
)
from hnn_core.network_builder import NetworkBuilder


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


@pytest.mark.parametrize(
    "fix_net_model", ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
)
def test_gid_assignment(fix_net_model, request):
    """Test that gids are assigned without overlap across ranks"""
    net_model = request.getfixturevalue(fix_net_model)
    net, net_customizations = net_model()
    inh_name_suffix = net_customizations["inh_name_suffix"]
    weights_ampa = {
        f"L2_{inh_name_suffix}": 1.0,
        "L2_pyramidal": 2.0,
        "L5_pyramidal": 3.0,
    }
    syn_delays = {
        f"L2_{inh_name_suffix}": 0.1,
        "L2_pyramidal": 0.2,
        "L5_pyramidal": 0.3,
    }

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

    def test_run_default(self, fix_net_neymotin_2020, fix_run_simulation):
        """Test consistency between default backend simulation and master"""
        global dpls_reduced_default
        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
        dpls_reduced_default = fix_run_simulation(net, tstop=40, backend=None)
        # test consistency across all parallel backends for multiple trials
        assert_raises(
            AssertionError,
            assert_array_equal,
            dpls_reduced_default[0].data["agg"],
            dpls_reduced_default[1].data["agg"],
        )

    def test_run_joblibbackend(self, fix_net_neymotin_2020, fix_run_simulation):
        """Test consistency between joblib backend simulation with master"""
        global dpls_reduced_default, dpls_reduced_joblib

        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
        dpls_reduced_joblib = fix_run_simulation(
            net, tstop=40, backend="joblib", n_jobs=2
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
        with MPIBackend(n_procs=None) as backend:  # Use all available cores
            assert backend.n_procs > 1

    @requires_mpi4py
    @requires_psutil
    def test_run_mpibackend(self, fix_net_neymotin_2020, fix_run_simulation):
        """Test running a MPIBackend on reduced model"""
        global dpls_reduced_default, dpls_reduced_mpi
        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
        # Use all available cores
        dpls_reduced_mpi = fix_run_simulation(net, tstop=40, backend="mpi")
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
    def test_terminate_mpibackend(self, fix_net_neymotin_2020):
        """Test terminating MPIBackend from thread"""
        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
        with MPIBackend(n_procs=None) as backend:  # Use all available cores
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
                    simulate_dipole(net, tstop=40, n_trials=2)

            event.set()
        expected_string = "Child process failed unexpectedly"
        assert expected_string in record[0].message.args[0]

    @requires_mpi4py
    @requires_psutil
    @pytest.mark.parametrize("use_hwthreading_if_found", [True, False])
    def test_run_mpibackend_oversubscribed(
        self, use_hwthreading_if_found, fix_default_params
    ):
        """Test running MPIBackend with oversubscribed number of procs"""
        params = fix_default_params
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
        fix_net_neymotin_2020,
    ):
        """Test running MPIBackend with oversubscribed number of procs"""
        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)

        n_procs = 2
        # Test that the network runs at all
        with MPIBackend(n_procs=n_procs) as backend:
            simulate_dipole(net, tstop=40, n_trials=2)

        [_, detected_hwthreading] = _determine_cores_hwthreading(
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
        )

        # Case 1 (default): Check that hwthreading turns on if needed
        override_hwthreading_option = None

        # Possibly needed to prevent MPIBackend failures to exit processes
        del net
        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)

        with MPIBackend(
            n_procs=n_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
            override_oversubscribe_option=override_oversubscribe_option,
            override_hwthreading_option=override_hwthreading_option,
        ) as backend:
            if detected_hwthreading:
                assert "--use-hwthread-cpus" in " ".join(backend.mpi_cmd)
            simulate_dipole(net, tstop=40, n_trials=2)

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
        net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)

        with MPIBackend(
            n_procs=n_procs,
            use_hwthreading_if_found=use_hwthreading_if_found,
            sensible_default_cores=sensible_default_cores,
            override_oversubscribe_option=override_oversubscribe_option,
            override_hwthreading_option=override_hwthreading_option,
        ) as backend:
            assert "--use-hwthread-cpus" not in " ".join(backend.mpi_cmd)
            simulate_dipole(net, tstop=40, n_trials=2)

    @pytest.mark.parametrize("backend", ["mpi", "joblib"])
    def test_compare_hnn_core(
        self, fix_net_neymotin_2020, fix_run_simulation, backend, n_jobs=1
    ):
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

        # TODO AES: CURRENTLY NEEDS LEGACY MODE
        net, _ = fix_net_neymotin_2020(legacy_mode=True, add_drives_from_params=True)
        dpls = fix_run_simulation(net, tstop=170, backend=backend)
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
def test_mpi_failure(fix_net_neymotin_2020, fix_run_simulation):
    """Test that an MPI failure is handled and messages are printed"""
    # this MPI parameter will cause a MPI job to fail
    environ["OMPI_MCA_btl"] = "self"

    net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
    with pytest.warns(UserWarning) as record:
        with io.StringIO() as buf, redirect_stdout(buf):
            with pytest.raises(RuntimeError, match="MPI simulation failed"):
                # Use all available cores
                fix_run_simulation(net, tstop=40, backend="mpi", postproc=False)
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


@requires_openmpi_package
def test_get_pip_openmpi_lib():
    """Test that only the real libmpi file in the 'openmpi' package is found"""
    mpi_lib = _get_pip_openmpi_lib()
    assert mpi_lib is not None
    # `_get_pip_openmpi_lib()` MUST find one or more of these files, which are the only
    # true MPI library files that NEURON needs.
    assert Path(mpi_lib).name in {
        "libmpi.so.40",
        "libmpi.so.12",
        "libmpi.40.dylib",
        "libmpi.12.dylib",
    }
    # `_get_pip_openmpi_lib()` must NOT find one or more of these files, which are also
    # present in the `openmpi` package depending on the OS. None of these files are the
    # "real" MPI library file that NEURON needs.
    assert Path(mpi_lib).name not in {
        "libmpi.so",
        "libmpi.dylib",
        "libmpi_mpifh.so.40",
        "libmpi.so.40.1.0",
        "libmpi.so.40.40.8",
    }
    assert Path(mpi_lib).is_file()


def test_get_pip_openmpi_lib_installed():
    """Test _get_pip_openmpi_lib against the actually-installed environment"""
    try:
        distribution("openmpi")
    except PackageNotFoundError:
        assert _get_pip_openmpi_lib() is None
    else:
        # If it's found to be installed, then the returned path of the library must
        # exist and be a file
        mpi_lib = _get_pip_openmpi_lib()
        assert mpi_lib is not None
        assert Path(mpi_lib).is_file()


# This tests that `_get_mpi_env` correctly sets the `MPI_LIB_NRN_PATH` environment
# variable based the state of the environment and which arguments are passed to
# it. There are four arguments that pytest parametrizes over:
#
# - `autoload_flag`: value of `autoload_mpi_library` argument to `_get_mpi_env`
# - `user_provided_lib`: value of `mpi_lib_path` argument to `_get_mpi_env`
# - `default_openmpi_lib`: value returned by `_get_pip_openmpi_lib` (which is mocked to
#     return a specific value)
# - `expected_returned_lib`: the expected value of the `MPI_LIB_NRN_PATH` environment
#     variable after `_get_mpi_env` is called
#
# Additionally, the test also sets the CALLER's environment variable `MPI_LIB_NRN_PATH`
# to a known value ("preexisting.so"). This is to test that `_get_mpi_env` does not
# modify the caller's environment itself.
#
# Think of everything except `expected_returned_lib` as "inputs" to the test, and
# `expected_returned_lib` as the "output" of the test.
@pytest.mark.parametrize(
    "autoload_flag, user_provided_lib, default_openmpi_lib, expected_returned_lib",
    [
        # Case when using default arguments if the `openmpi` package is installed:
        # - autoload_flag = True
        # - user_provided_lib = None (no custom library path provided)
        # - default_openmpi_lib = mock value of the installed library file "pip.so"
        # - expected_returned_lib = _get_mpi_env should automatically set the
        #     environment variable to `default_openmpi_lib`: "pip.so"
        pytest.param(True, None, "pip.so", "pip.so", id="default-installed"),
        # Case when using default arguments if the `openmpi` package is NOT installed:
        # - autoload_flag = True
        # - user_provided_lib = None (no custom library path provided)
        # - default_openmpi_lib = None (not installed)
        # - expected_returned_lib = the environment variable should remain UNCHANGED
        #     from its preexisting value "preexisting.so"
        pytest.param(True, None, None, "preexisting.so", id="default-not-installed"),
        # Case when autoload is off, and no user provided library path is given:
        # - autoload_flag = False
        # - user_provided_lib = None (no custom library path provided)
        # - default_openmpi_lib = mock value of the installed library file "pip.so"
        # - expected_returned_lib = the environment variable should remain UNCHANGED
        #     from its preexisting value "preexisting.so"
        pytest.param(False, None, "pip.so", "preexisting.so", id="no-autoload"),
        # Case when autoload is off, and user provided library path has been given:
        # - autoload_flag = False
        # - user_provided_lib = "user.so" (custom library path provided)
        # - default_openmpi_lib = mock value of the installed library file "pip.so"
        # - expected_returned_lib = _get_mpi_env should automatically set the
        #     environment variable to `user_provided_lib`: "user.so"
        pytest.param(False, "user.so", "pip.so", "user.so", id="no-autoload-user"),
        # Case when autoload is ON, and user provided library path has been given. In
        # this case, the user-provided library path should take precedence over the
        # autoloaded library path:
        # - autoload_flag = True
        # - user_provided_lib = "user.so" (custom library path provided)
        # - default_openmpi_lib = mock value of the installed library file "pip.so"
        # - expected_returned_lib = _get_mpi_env should automatically set the
        #     environment variable to `user_provided_lib`: "user.so"
        pytest.param(True, Path("user.so"), "pip.so", "user.so", id="user-wins"),
    ],
)
def test_get_mpi_env_lib_path(
    monkeypatch,
    autoload_flag,
    user_provided_lib,
    default_openmpi_lib,
    expected_returned_lib,
):
    """Test which MPI library _get_mpi_env points MPI_LIB_NRN_PATH to across arguments"""
    # Set the caller's environment, in order to make sure that `_get_mpi_env` later does
    # not modify it
    monkeypatch.setenv("MPI_LIB_NRN_PATH", "preexisting.so")
    # Overwrite the `_get_pip_openmpi_lib` function to return a specific library path
    # for testing purposes.
    monkeypatch.setattr(
        parallel_backends, "_get_pip_openmpi_lib", lambda: default_openmpi_lib
    )

    env = _get_mpi_env(
        autoload_mpi_library=autoload_flag, mpi_lib_path=user_provided_lib
    )

    assert env["MPI_LIB_NRN_PATH"] == expected_returned_lib
    # Check that the caller's own environment was not modified
    assert environ["MPI_LIB_NRN_PATH"] == "preexisting.so"


@requires_mpi4py
@requires_psutil
def test_mpibackend_lib_path(monkeypatch, tmp_path):
    """Test that 'mpi_lib_path' must exist and reaches MPI as an absolute path

    Note that this test does not actually depend on whether the `openmpi` package is
    installed or not, but does require that the `mpi4py` and `psutil` packages are
    installed.
    """
    # "nonexistent.so" tests a path that does not exist, and `tmp_path` tests a path
    # that exists but is a directory instead of a file.
    for bad_path in ["nonexistent.so", tmp_path]:
        with pytest.raises(FileNotFoundError, match="'mpi_lib_path' not found"):
            MPIBackend(mpi_lib_path=bad_path)

    # Capture the env that MPI would be launched with, but stop before launching it
    def _fake_run_subprocess(env, **kwargs):
        raise RuntimeError(f"MPI_LIB_NRN_PATH={env['MPI_LIB_NRN_PATH']}")

    monkeypatch.setattr(parallel_backends, "run_subprocess", _fake_run_subprocess)

    # Create a fake MPI library file in the temporary directory, in order to make sure
    # that MPIBackend is accessing the correct path to the library.
    (tmp_path / "libmpi.so.40").touch()

    # Setup our network and backend, pointing the backend to the fake MPI library file.
    net = neymotin_2020_model(mesh_shape=(3, 3))
    backend = MPIBackend(n_procs=2, mpi_lib_path=(tmp_path / "libmpi.so.40"))

    # Test that at simulation time, the MPIBackend can see and is using the correct path
    # to the MPI library file, based on its internal `env` setting. We do this by
    # hacking the `run_subprocess` using the monkeypatch above.
    expected = re.escape(f"MPI_LIB_NRN_PATH={tmp_path / 'libmpi.so.40'}")
    with pytest.raises(RuntimeError, match=f"^{expected}$"):
        backend.simulate(net, tstop=1, dt=0.025, n_trials=1)
