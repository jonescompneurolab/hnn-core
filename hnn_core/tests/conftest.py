""" Example from pytest documentation

https://pytest.org/en/stable/example/simple.html#incremental-testing-test-steps
"""

from typing import Dict, Tuple
import pytest

import os.path as op
import hnn_core
from hnn_core import read_params, default_network, simulate_dipole
from hnn_core import MPIBackend, JoblibBackend

# store history of failures per test class name and per index in parametrize
# (if parametrize used)
_test_failed_incremental: Dict[str, Dict[Tuple[int, ...], str]] = {}


def pytest_runtest_makereport(item, call):

    if "incremental" in item.keywords:
        # incremental marker is used

        # The following condition was modifed from the example linked above.
        # We don't want to step out of the incremental testing block if
        # a previous test was marked "Skipped". For instance if MPI tests
        # are skipped because mpi4py is not installed, still continue with
        # all other tests that do not require mpi4py
        if call.excinfo is not None and not call.excinfo.typename == "Skipped":
            # the test has failed, but was not skiped

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
            test_name = _test_failed_incremental[cls_name].get(
                parametrize_index, None)
            # if name found, test has failed for the combination of class name
            # and test name
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))


@pytest.fixture(scope='module')
def run_hnn_core_fixture():
    def _run_hnn_core_fixture(backend=None, n_procs=None, n_jobs=1,
                              reduced=False, record_vsoma=False,
                              record_isoma=False, postproc=True):
        hnn_core_root = op.dirname(hnn_core.__file__)

        # default params
        params_fname = op.join(hnn_core_root, 'param', 'default.json')
        params = read_params(params_fname)

        if reduced:
            params.update({'N_pyr_x': 3,
                           'N_pyr_y': 3,
                           'tstop': 40,
                           't_evprox_1': 5,
                           't_evdist_1': 10,
                           't_evprox_2': 20,
                           'N_trials': 2})
        net = default_network(params, add_drives_from_params=True)

        # number of trials simulated
        for drive in net.external_drives.values():
            assert len(drive['events']) == params['N_trials']

        if backend == 'mpi':
            with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
                dpls = simulate_dipole(net, record_vsoma=record_isoma,
                                       record_isoma=record_vsoma,
                                       postproc=postproc)
        elif backend == 'joblib':
            with JoblibBackend(n_jobs=n_jobs):
                dpls = simulate_dipole(net, record_vsoma=record_isoma,
                                       record_isoma=record_vsoma,
                                       postproc=postproc)
        else:
            dpls = simulate_dipole(net, record_vsoma=record_isoma,
                                   record_isoma=record_vsoma,
                                   postproc=postproc)

        return dpls, net
    return _run_hnn_core_fixture
