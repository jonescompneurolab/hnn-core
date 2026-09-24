"""Example from pytest documentation

https://pytest.org/en/stable/example/simple.html#incremental-testing-test-steps
"""

from typing import Dict, Tuple
import copy
import pytest
import pickle

from pathlib import Path
import hnn_core
from hnn_core import (
    read_params,
    calcium_model,
    duecker_ET_model,
    neymotin_2020_model,
    simulate_dipole,
)
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
def fix_net_duecker_ET():
    """Test fixture for the Duecker ET model network.

    Note that the argument `legacy_mode` is unused, since it is only present for API
    equality with `fix_net_neymotin_2020`.
    """

    def _fix_net_duecker_ET(
        add_drives_from_params=False,
        legacy_mode=False,
        reduced=False,
        electrode_array=None,
    ):
        if reduced:
            mesh_shape = (3, 3)
            # Shorten when the drives start, since we usually use a shorter simulation
            # when using a reduced network.
            prox1_mu = 5
            dist1_mu = 10
            prox2_mu = 20
        else:
            mesh_shape = (10, 10)
            prox1_mu = 18
            dist1_mu = 62
            prox2_mu = 100

        net = duecker_ET_model(
            mesh_shape=mesh_shape,
        )

        if add_drives_from_params:
            weights_ampa_p1 = {
                "L2_inhibitory": 0.01,
                "L2_pyramidal": 0.015,
                "L5_inhibitory": 0.0,
                "L5_pyramidal": 0.03,
            }
            weights_nmda_p1 = {
                "L2_inhibitory": 0.01,
                "L2_pyramidal": 0.05,
                "L5_inhibitory": 0.0,
                "L5_pyramidal": 0.025,
            }
            synaptic_delays_prox = {
                "L2_inhibitory": 0.1,
                "L2_pyramidal": 0.1,
                "L5_inhibitory": 1,
                "L5_pyramidal": 1,
            }

            net.add_evoked_drive(
                "evprox1",
                mu=prox1_mu,
                sigma=2.5,
                numspikes=1,
                weights_ampa=weights_ampa_p1,
                weights_nmda=weights_nmda_p1,
                location="proximal",
                synaptic_delays=synaptic_delays_prox,
            )

            weights_ampa_d1 = {
                "L2_inhibitory": 0.005,
                "L2_pyramidal": 0.01,
                "L5_pyramidal": 1.0,
            }
            weights_nmda_d1 = {
                "L2_inhibitory": 0.0,
                "L2_pyramidal": 0.01,
                "L5_pyramidal": 1.0,
            }
            synaptic_delays_dist = {
                "L2_inhibitory": 0.1,
                "L2_pyramidal": 0.1,
                "L5_pyramidal": 0.1,
            }

            net.add_evoked_drive(
                "evdist1",
                mu=dist1_mu,
                sigma=5,
                numspikes=2,
                weights_ampa=weights_ampa_d1,
                weights_nmda=weights_nmda_d1,
                location="distal",
                synaptic_delays=synaptic_delays_dist,
            )

            weights_ampa_p2 = {
                "L2_inhibitory": 0.01,
                "L2_pyramidal": 0.3,
                "L5_inhibitory": 0.001,
                "L5_pyramidal": 0.3,
            }
            weights_nmda_p2 = {
                "L2_inhibitory": 0.01,
                "L2_pyramidal": 0.2,
                "L5_inhibitory": 0.001,
                "L5_pyramidal": 0.2,
            }
            synaptic_delays_prox = {
                "L2_inhibitory": 0.1,
                "L2_pyramidal": 0.1,
                "L5_inhibitory": 1.0,
                "L5_pyramidal": 1.0,
            }
            net.add_evoked_drive(
                "evprox2",
                mu=prox2_mu,
                sigma=15,
                numspikes=1,
                weights_ampa=weights_ampa_p2,
                weights_nmda=weights_nmda_p2,
                location="proximal",
                synaptic_delays=synaptic_delays_prox,
            )

        net.set_cell_positions(inplane_distance=30.0)
        if electrode_array is not None:
            for name, positions in electrode_array.items():
                net.add_electrode_array(name, positions)

        return net

    return _fix_net_duecker_ET


@pytest.fixture(scope="module")
def fix_net_neymotin_2020():
    """Test fixture for the Neymotin 2020 model network.

    TODO Docstring coming soon! UNDER CONSTRUCTION <construction-beaver.gif>
    """

    def _fix_net_neymotin_2020(
        add_drives_from_params=False,
        legacy_mode=False,
        reduced=False,
        electrode_array=None,
        featureful_reduced_network=False,
    ):
        # default params
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)

        if featureful_reduced_network and (
            legacy_mode or reduced or electrode_array is not None
        ):
            raise ValueError(
                "featureful_reduced_network cannot be used with legacy_mode, reduced, "
                "or electrode_array arguments."
            )

        if not featureful_reduced_network:
            if reduced:
                mesh_shape = (3, 3)
                # NOTE: `run_hnn_core_fixture` with `reduced=True` originally set:
                # - trials to 2 using the `Network` object (instead of at simulation)
                # - set simulation time to 40 ms, and
                # - disabled legacy_mode
                # Trials and simulation time are now only set at simulation time, and legacy
                # mode is a regular argument.
                # TODO AES: Use the API for this, NOT params!
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

        # Formerly called the network at
        # `hnn_core/tests/assets/neymotin2020_3x3_drives.json`
        elif featureful_reduced_network:
            net = neymotin_2020_model(
                params=None,
                add_drives_from_params=True,
                legacy_mode=False,
                mesh_shape=(3, 3),
            )
            # Adding bias
            tonic_bias = {
                "L2_pyramidal": 1.0,
                "L5_pyramidal": 0.0,
                "L2_basket": 0.0,
                "L5_basket": 0.0,
            }
            net.add_tonic_bias(amplitude=tonic_bias)

            # Add drives
            location = "proximal"
            burst_std = 20
            weights_ampa_p = {
                "L2_pyramidal": 5.4e-5,
                "L5_pyramidal": 5.4e-5,
                "L2_basket": 0.0,
                "L5_basket": 0.0,
            }
            weights_nmda_p = {
                "L2_pyramidal": 0.0,
                "L5_pyramidal": 0.0,
                "L2_basket": 0.0,
                "L5_basket": 0.0,
            }
            syn_delays_p = {
                "L2_pyramidal": 0.1,
                "L5_pyramidal": 1.0,
                "L2_basket": 0.0,
                "L5_basket": 0.0,
            }
            net.add_bursty_drive(
                "alpha_prox",
                tstart=1.0,
                burst_rate=10,
                burst_std=burst_std,
                numspikes=2,
                spike_isi=10,
                n_drive_cells=10,
                location=location,
                weights_ampa=weights_ampa_p,
                weights_nmda=weights_nmda_p,
                synaptic_delays=syn_delays_p,
                event_seed=284,
            )

            weights_ampa = {
                "L2_pyramidal": 0.0008,
                "L5_pyramidal": 0.0075,
                "L2_basket": 0.0,
                "L5_basket": 0.0,
            }
            synaptic_delays = {
                "L2_pyramidal": 0.1,
                "L5_pyramidal": 1.0,
                "L2_basket": 0.0,
                "L5_basket": 0.0,
            }
            rate_constant = {
                "L2_pyramidal": 140.0,
                "L5_pyramidal": 40.0,
                "L2_basket": 40.0,
                "L5_basket": 40.0,
            }
            net.add_poisson_drive(
                "poisson",
                rate_constant=rate_constant,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda_p,
                location="proximal",
                synaptic_delays=synaptic_delays,
                event_seed=1349,
            )

            # Adding electrode arrays
            electrode_pos = (1, 2, 3)
            net.add_electrode_array("el1", electrode_pos)
            electrode_pos = [(1, 2, 3), (-1, -2, -3)]
            net.add_electrode_array("arr1", electrode_pos)

        return net

    return _fix_net_neymotin_2020


@pytest.fixture(scope="module")
def fix_load_featureful_tmp_path(tmp_path_factory, fix_net_neymotin_2020):
    """Load the featureful reduced Neymotin 2020 network from the fixture."""
    net = fix_net_neymotin_2020(featureful_reduced_network=True)
    net_path = (
        tmp_path_factory.mktemp("network") / "neymotin_2020_featureful_reduced.json"
    )
    net.write_configuration(net_path, overwrite=True)
    return net_path


@pytest.fixture(scope="module")
def fix_net_calcium():
    """Test fixture for the "Calcium" model network.

    TODO Docstring coming soon! UNDER CONSTRUCTION <construction-beaver.gif>
    """

    def _fix_net_calcium(
        add_drives_from_params=False,
        legacy_mode=False,
        reduced=False,
        electrode_array=None,
    ):
        if reduced:
            mesh_shape = (3, 3)
        else:
            mesh_shape = (10, 10)
        # Legacy mode necessary for exact dipole comparison test
        net = calcium_model(
            add_drives_from_params=add_drives_from_params,
            legacy_mode=legacy_mode,
            mesh_shape=mesh_shape,
        )
        if electrode_array is not None:
            for name, positions in electrode_array.items():
                net.add_electrode_array(name, positions)

        return net

    return _fix_net_calcium


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
def _base_simulation_cached():
    """Adds bursty drives and simulates once per network model and variation"""
    cache = {}
    # AMPA weights of the bursty drives for each variation
    variation_weights_ampa = {
        "yes_spikes": {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0},
        "no_spikes": {"L2_pyramidal": 5.4e-5, "L5_pyramidal": 5.4e-5},
    }

    def _get_simulation(net_model_name, net_model, variation):
        key = (net_model_name, variation)
        if key not in cache:
            net = net_model(reduced=True)
            # Account for Duecker name variations
            inh_name = "basket" if "L2_basket" in net.cell_types else "inhibitory"
            weights_ampa = variation_weights_ampa[variation]
            syn_delays = {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0}
            net.add_bursty_drive(
                "beta_prox",
                tstart=0.0,
                burst_rate=25,
                burst_std=5,
                numspikes=1,
                spike_isi=0,
                n_drive_cells=11,
                location="proximal",
                weights_ampa=weights_ampa,
                synaptic_delays=syn_delays,
                event_seed=14,
            )

            net.add_bursty_drive(
                "beta_dist",
                tstart=0.0,
                burst_rate=25,
                burst_std=5,
                numspikes=1,
                spike_isi=0,
                n_drive_cells=11,
                location="distal",
                weights_ampa=weights_ampa,
                synaptic_delays=syn_delays,
                event_seed=14,
            )
            dpls = simulate_dipole(net, tstop=100.0, n_trials=2, record_vsec="all")
            cache[key] = (net, dpls, inh_name)
        # Deepcopy so each caller gets its own net and dpls
        return copy.deepcopy(cache[key])

    return _get_simulation


@pytest.fixture(
    scope="function",
    params=["fix_net_neymotin_2020", "fix_net_duecker_ET"],
)
def fix_use_cached_sim_yes_spikes(_base_simulation_cached, request):
    """Copy of the cached simulation, for spike visualization tests"""
    net_model_name = request.param
    net_model = request.getfixturevalue(net_model_name)
    net, dpls, inh_name = _base_simulation_cached(
        net_model_name, net_model, variation="yes_spikes"
    )
    return net, dpls, inh_name


@pytest.fixture(
    scope="function",
    params=["fix_net_neymotin_2020", "fix_net_duecker_ET"],
)
def fix_use_cached_sim_no_spikes(_base_simulation_cached, request):
    """Copy of the cached simulation, for spike visualization tests"""
    net_model_name = request.param
    net_model = request.getfixturevalue(net_model_name)
    net, dpls, inh_name = _base_simulation_cached(
        net_model_name, net_model, variation="no_spikes"
    )
    return net, dpls, inh_name


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
