"""Example from pytest documentation

https://pytest.org/en/stable/example/simple.html#incremental-testing-test-steps
"""

from typing import Dict, Tuple
import pytest
import pickle

from pathlib import Path
import hnn_core
from hnn_core import read_params, neymotin_2020_model, simulate_dipole
from hnn_core import MPIBackend, JoblibBackend

# store history of failures per test class name and per index in parametrize
# (if parametrize used)
_test_failed_incremental: Dict[str, Dict[Tuple[int, ...], str]] = {}

hnn_core_root = Path(hnn_core.__file__).parent


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        # incremental marker is used

        # The following condition was modified from the example linked above.
        # We don't want to step out of the incremental testing block if
        # a previous test was marked "Skipped". For instance if MPI tests
        # are skipped because mpi4py is not installed, still continue with
        # all other tests that do not require mpi4py
        if call.excinfo is not None and not call.excinfo.typename == "Skipped":
            # the test has failed, but was not skipped

            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in
            # combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the
            # failed test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in
            # combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the first test function to fail for this
            # class name and index
            test_name = _test_failed_incremental[cls_name].get(parametrize_index, None)
            # if name found, test has failed for the combination of class name
            # and test name
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))


@pytest.fixture(scope="module")
def fix_net_neymotin_2020():
    def _fix_net_neymotin_2020(
        add_drives_from_params=True,
        legacy_mode=False,
        reduced=False,
        electrode_array=None,
    ):
        # default params
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)

        if reduced:
            mesh_shape = (3, 3)
            # NOTE: `run_hnn_core_fixture` with `reduced=True` originally set:
            # - trials to 2 using the `Network` object (instead of at simulation)
            # - set simulation time to 40 ms, and
            # - disabled legacy_mode
            # Trials and simulation time are now only set at simulation time, and legacy
            # mode is a regular argument.
            params.update({"t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20})
        else:
            mesh_shape = (10, 10)
        # Legacy mode necessary for exact dipole comparison test
        net = neymotin_2020_model(
            params,
            add_drives_from_params=add_drives_from_params,
            legacy_mode=legacy_mode,
            mesh_shape=mesh_shape,
        )
        if electrode_array is not None:
            for name, positions in electrode_array.items():
                net.add_electrode_array(name, positions)

        return net

    return _fix_net_neymotin_2020


@pytest.fixture(scope="module")
def fix_run_simulation():
    def _fix_run_simulation(
        net,
        tstop,
        dt=0.025,
        n_trials=2,  # default is 2!!!
        record_vsec=False,
        record_isec=False,
        record_ca=False,
        postproc=False,
        verbose=True,
        bsl_cor=None,
        backend=None,
        n_procs=None,
        n_jobs=1,
    ):
        if backend == "mpi":
            with MPIBackend(n_procs=n_procs, mpi_cmd="mpiexec"):
                dpls = simulate_dipole(
                    net,
                    tstop=tstop,
                    dt=dt,
                    n_trials=n_trials,
                    record_vsec=record_vsec,
                    record_isec=record_isec,
                    record_ca=record_ca,
                    postproc=postproc,
                    verbose=verbose,
                    bsl_cor=bsl_cor,
                )
        elif backend == "joblib":
            with JoblibBackend(n_jobs=n_jobs):
                dpls = simulate_dipole(
                    net,
                    tstop=tstop,
                    dt=dt,
                    n_trials=n_trials,
                    record_vsec=record_vsec,
                    record_isec=record_isec,
                    record_ca=record_ca,
                    postproc=postproc,
                    verbose=verbose,
                    bsl_cor=bsl_cor,
                )
        else:
            dpls = simulate_dipole(
                net,
                tstop=tstop,
                dt=dt,
                n_trials=n_trials,
                record_vsec=record_vsec,
                record_isec=record_isec,
                record_ca=record_ca,
                postproc=postproc,
                verbose=verbose,
                bsl_cor=bsl_cor,
            )

        # check that the network object is picklable after the simulation
        pickle.dumps(net)

        # number of trials simulated
        for drive in net.external_drives.values():
            # In the old `run_hnn_core_fixture`, simulated trials were compared against
            # the Network object's `_params["N_trials"]` attribute. However, for the
            # sake of eventually moving past usage of `params`, trials will now be
            # compared to the number provided by the argument.
            assert len(drive["events"]) == n_trials

        return dpls, net

    return _fix_run_simulation


@pytest.fixture(scope="module")
def fix_default_params():
    """Return the loaded default "flat JSON" parameters for the Neymotin 2020 (aka Jones 2009) model."""
    params_fname = hnn_core_root / "param" / "default.json"
    return read_params(params_fname)


@pytest.fixture
def network_default(fix_default_params):
    """Default Neymotin 2020 (aka Jones 2009) network with drives."""
    return neymotin_2020_model(fix_default_params, add_drives_from_params=True)


@pytest.fixture
def network_no_drives(fix_default_params):
    """Default Neymotin 2020 (aka Jones 2009) network without external drives."""
    return neymotin_2020_model(fix_default_params, add_drives_from_params=False)


@pytest.fixture
def network_small(fix_default_params):
    """Small network (1x1 mesh) for faster tests."""
    return neymotin_2020_model(
        fix_default_params,
        add_drives_from_params=True,
        mesh_shape=(1, 1),
    )


@pytest.fixture
def network_3x3(fix_default_params):
    """3x3 mesh network used for testing larger network configurations."""
    return neymotin_2020_model(
        fix_default_params,
        add_drives_from_params=True,
        mesh_shape=(3, 3),
    )
