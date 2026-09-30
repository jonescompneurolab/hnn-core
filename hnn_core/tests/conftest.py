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
    law_2021_model,
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


@pytest.fixture(scope="session")
def fix_net_duecker_ET():
    """Network fixture for tests of the Duecker ET model network (``duecker_ET_model``).

    You can use this and other network fixtures in one of two ways. All network fixtures
    can be used in the following ways:

    1. If you only want to use this network, and do not want to parametrize the test
    across multiple network models, then you can use this like a fixture that accepts
    arguments. For example, in ``test_dipole.py::test_dipole_simulation``, you include
    the fixture in the test's arguments:

    ```
    def test_dipole_simulation(fix_net_neymotin_2020, fix_default_params):
    ```

    and then use the fixture similarly to a regular network models function:

    ```
    net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
    ```

    2. If you want to use this network fixture while parametrizing a test to use other
    network fixtures as well, such as in
    ``test_extracellular.py::test_transmembrane_currents``, then you have to do four
    separate things:

    2.1. Add this fixture as an entry to the ``@pytest.mark.parametrize`` decorator such
    that this network fixture is a value in the list of values given to the "parameter"
    variable. For example:

    ```
    @pytest.mark.parametrize(
        "fix_net_model", ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
    )
    ```

    In the above, ``fix_net_model`` will be a variable that you pass to the argument list
    of the test, and the entries in the list are the values that that parameter will
    take, as the test is re-run once per parameter set.

    2.2. Add both the built-in ``request`` fixture along with the above "parameter"
    variable to your test function's arguments, e.g.

    ```
    def test_transmembrane_currents(fix_net_model, request):
    ```

    2.2. Create the callable network function by requesting the factory fixture's value,
    e.g.
    ```
    net_model = request.getfixturevalue(fix_net_model)
    ```

    (Note that this does NOT create the actual network, it only creates the
    *function/callable* to create the network).

    2.3. Finally, use your new callable to create the network, using the Parameters
    described below, e.g.

    ```
    net, inh_name = net_model(add_drives_from_params=True, reduced=True)
    ```

    This is what actually deploys the appropriate fixture and builds your model, using
    the arguments provided below.

    There are many examples of this pattern in the tests, so if you are unsure how to
    use this, see existing tests.

    Parameters
    ----------
    add_drives_from_params : bool, default=False
        If True, add three evoked drives (``evprox1``, ``evdist1``, and ``evprox2``)
        that mirror the default ERP drives of ``neymotin_2020_model``, except that the
        cell type names have been updated to match the Duecker model.
    legacy_mode : bool, default=False
        Unused. Only present for API equality with other network fixtures, so that all
        fixtures can be called interchangeably in parametrized tests.
    mesh_shape : tuple of int | None, default=None
        Shape of the cell grid. If None (default), a ``mesh_shape`` of ``(10, 10)`` is
        used. Incompatible with ``reduced=True``.
    reduced : bool, default=False
        If True, use a small ``(3, 3)`` mesh and shift the evoked drive times earlier
        (``evprox1``: 5 ms, ``evdist1``: 10 ms, ``evprox2``: 20 ms) so they fit within
        the shorter simulations typically used with a reduced network. (Otherwise the
        drive times are 18, 62, and 100 ms). Incompatible with ``mesh_shape`` of value
        other than None.
    electrode_array : dict | None, default=None
        Mapping of electrode array names to their lists of positions, each passed
        to ``net.add_electrode_array``.

    Returns
    -------
    net : Network object
        The Duecker ET model network.
    inh_name : "inhibitory"
        Name suffix of the inhibitory cell types, which is ``"inhibitory"`` for the
        Duecker model. Used to account for naming differences between model versions.

    Raises
    ------
    ValueError
        If both ``reduced=True`` and ``mesh_shape`` are given.
    """

    def _fix_net_duecker_ET(
        add_drives_from_params=False,
        legacy_mode=False,
        mesh_shape=None,
        reduced=False,
        electrode_array=None,
    ):
        if reduced and mesh_shape:
            raise ValueError(
                "Cannot specify both `reduced=True` and `mesh_shape` argument."
            )

        # Account for Duecker name variations
        inh_name = "inhibitory"

        if reduced:
            mesh_shape = (3, 3)
            # Shorten when the drives start, since we usually use a shorter simulation
            # when using a reduced network.
            prox1_mu = 5
            dist1_mu = 10
            prox2_mu = 20
        else:
            if mesh_shape is not None:
                pass  # Use the provided mesh_shape
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

        if electrode_array is not None:
            for name, positions in electrode_array.items():
                net.add_electrode_array(name, positions)

        return net, inh_name

    return _fix_net_duecker_ET


@pytest.fixture(scope="session")
def fix_net_neymotin_2020():
    """Network fixture for tests of the Neymotin 2020 model network (``neymotin_2020_model``).

    You can use this and other network fixtures in one of two ways. All network fixtures
    can be used in the following ways:

    1. If you only want to use this network, and do not want to parametrize the test
    across multiple network models, then you can use this like a fixture that accepts
    arguments. For example, in ``test_dipole.py::test_dipole_simulation``, you include
    the fixture in the test's arguments:

    ```
    def test_dipole_simulation(fix_net_neymotin_2020, fix_default_params):
    ```

    and then use the fixture similarly to a regular network models function:

    ```
    net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
    ```

    2. If you want to use this network fixture while parametrizing a test to use other
    network fixtures as well, such as in
    ``test_extracellular.py::test_transmembrane_currents``, then you have to do four
    separate things:

    2.1. Add this fixture as an entry to the ``@pytest.mark.parametrize`` decorator such
    that this network fixture is a value in the list of values given to the "parameter"
    variable. For example:

    ```
    @pytest.mark.parametrize(
        "fix_net_model", ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
    )
    ```

    In the above, ``fix_net_model`` will be a variable that you pass to the argument list
    of the test, and the entries in the list are the values that that parameter will
    take, as the test is re-run once per parameter set.

    2.2. Add both the built-in ``request`` fixture along with the above "parameter"
    variable to your test function's arguments, e.g.

    ```
    def test_transmembrane_currents(fix_net_model, request):
    ```

    2.2. Create the callable network function by requesting the factory fixture's value,
    e.g.
    ```
    net_model = request.getfixturevalue(fix_net_model)
    ```

    (Note that this does NOT create the actual network, it only creates the
    *function/callable* to create the network).

    2.3. Finally, use your new callable to create the network, using the Parameters
    described below, e.g.

    ```
    net, inh_name = net_model(add_drives_from_params=True, reduced=True)
    ```

    This is what actually deploys the appropriate fixture and builds your model, using
    the arguments provided below.

    There are many examples of this pattern in the tests, so if you are unsure how to
    use this, see existing tests.

    Parameters
    ----------
    add_drives_from_params : bool, default=False
        If True, pass ``add_drives_from_params=True`` to the ``neymotin_2020_model``
        call in order to add the three canonical evoked drives (``evprox1``,
        ``evdist1``, and ``evprox2``). Incompatible with
        ``featureful_reduced_network=True``.
    legacy_mode : bool, default=False
        If True, pass ``add_drives_from_params=True`` to the ``neymotin_2020_model``
        call for testing deprecated legacy behavior. Incompatible with
        ``featureful_reduced_network=True``.
    mesh_shape : tuple of int | None, default=None
        Shape of the cell grid. If None (default), a ``mesh_shape`` of ``(10, 10)`` is
        used. Incompatible with ``reduced=True`` or ``featureful_reduced_network=True``.
    reduced : bool, default=False
        If True, use a small ``(3, 3)`` mesh and shift the evoked drive times earlier
        (``evprox1``: 5 ms, ``evdist1``: 10 ms, ``evprox2``: 20 ms) so they fit within
        the shorter simulations typically used with a reduced network. (Otherwise the
        drive times are 18, 62, and 100 ms). Incompatible with ``mesh_shape`` of value
        other than None or ``featureful_reduced_network=True``.
    electrode_array : dict | None, default=None
        Mapping of electrode array names to their lists of positions, each passed
        to ``net.add_electrode_array``. Incompatible with
        ``featureful_reduced_network=True``.
    featureful_reduced_network : bool, default=False
        Incompatible with all other arguments. If True, use a ``reduced`` network with a
        small ``(3, 3)`` mesh, ``add_drives_from_params=True``, a bias, multiple
        electrode arrays, and a bursty and Poisson drive. This is called "featureful"
        because it includes all types of Network features, in order to test that all
        features are correctly serialized, written, deserialized, and loaded, including
        via both the API and the GUI. This is identical to the network that was formerly
        stored at ``hnn_core/tests/assets/neymotin2020_3x3_drives.json``.

    Returns
    -------
    net : Network object
        The Neymotin 2020 model network.
    inh_name : "basket"
        Name suffix of the inhibitory cell types, which is ``"basket"`` for the
        non-Duecker models.

    Raises
    ------
    ValueError
        If both ``reduced=True`` and ``mesh_shape`` are given.
    ValueError
        If ``featureful_reduced_network=True`` is given along with any other argument.
    """

    def _fix_net_neymotin_2020(
        add_drives_from_params=False,
        legacy_mode=False,
        mesh_shape=None,
        reduced=False,
        electrode_array=None,
        featureful_reduced_network=False,
    ):
        if featureful_reduced_network and (
            add_drives_from_params
            or legacy_mode
            or (mesh_shape is not None)
            or reduced
            or (electrode_array is not None)
        ):
            raise ValueError(
                "featureful_reduced_network cannot be used with legacy_mode, reduced, "
                "or electrode_array arguments."
            )

        # default params
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
        # Account for Duecker name variations
        inh_name = "basket"

        if not featureful_reduced_network:
            if reduced and mesh_shape:
                raise ValueError(
                    "Cannot specify both `reduced=True` and `mesh_shape` argument."
                )
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
            elif mesh_shape is not None:
                pass  # Use the provided mesh_shape
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

        return net, inh_name

    return _fix_net_neymotin_2020


@pytest.fixture(scope="module")
def fix_load_featureful_tmp_path(tmp_path_factory, fix_net_neymotin_2020):
    """Save a "featureful" Neymotin 2020 network fixture to file, and return the path.

    This is used to test that the network can be serialized and deserialized correctly by saving a copy of

    ```
    fix_net_neymotin_2020(featureful_reduced_network=True)
    ```

    to a temporary file and returning the path to that file. The temporary file is
    deleted after the test session ends. This allows both the GUI and API to test
    loading the network from file, but without having to carry around a permanent copy
    of the network. This functionality replaces the previously stored network at
    ``hnn_core/tests/assets/neymotin2020_3x3_drives.json``.

    Parameters
    ----------
    This takes no user-provided arguments, and instead requires only other fixtures that
    are automatically provided via the argument list.

    Returns
    -------
    net_path : Path
        Path to the temporary file containing the serialized "featureful" Neymotin 2020
        network.
    """
    net, _ = fix_net_neymotin_2020(featureful_reduced_network=True)
    net_path = (
        tmp_path_factory.mktemp("network") / "neymotin_2020_featureful_reduced.json"
    )
    net.write_configuration(net_path, overwrite=True)
    return net_path


@pytest.fixture(scope="session")
def fix_net_calcium():
    """Network fixture for tests of the "Calcium" model network (``calcium_model``).

    You can use this and other network fixtures in one of two ways. All network fixtures
    can be used in the following ways:

    1. If you only want to use this network, and do not want to parametrize the test
    across multiple network models, then you can use this like a fixture that accepts
    arguments. For example, in ``test_dipole.py::test_dipole_simulation``, you include
    the fixture in the test's arguments:

    ```
    def test_dipole_simulation(fix_net_neymotin_2020, fix_default_params):
    ```

    and then use the fixture similarly to a regular network models function:

    ```
    net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
    ```

    2. If you want to use this network fixture while parametrizing a test to use other
    network fixtures as well, such as in
    ``test_extracellular.py::test_transmembrane_currents``, then you have to do four
    separate things:

    2.1. Add this fixture as an entry to the ``@pytest.mark.parametrize`` decorator such
    that this network fixture is a value in the list of values given to the "parameter"
    variable. For example:

    ```
    @pytest.mark.parametrize(
        "fix_net_model", ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
    )
    ```

    In the above, ``fix_net_model`` will be a variable that you pass to the argument list
    of the test, and the entries in the list are the values that that parameter will
    take, as the test is re-run once per parameter set.

    2.2. Add both the built-in ``request`` fixture along with the above "parameter"
    variable to your test function's arguments, e.g.

    ```
    def test_transmembrane_currents(fix_net_model, request):
    ```

    2.2. Create the callable network function by requesting the factory fixture's value,
    e.g.
    ```
    net_model = request.getfixturevalue(fix_net_model)
    ```

    (Note that this does NOT create the actual network, it only creates the
    *function/callable* to create the network).

    2.3. Finally, use your new callable to create the network, using the Parameters
    described below, e.g.

    ```
    net, inh_name = net_model(add_drives_from_params=True, reduced=True)
    ```

    This is what actually deploys the appropriate fixture and builds your model, using
    the arguments provided below.

    There are many examples of this pattern in the tests, so if you are unsure how to
    use this, see existing tests.

    Parameters
    ----------
    add_drives_from_params : bool, default=False
        If True, pass ``add_drives_from_params=True`` to the ``calcium_model`` call in
        order to add the three canonical evoked drives (``evprox1``, ``evdist1``, and
        ``evprox2``).
    legacy_mode : bool, default=False
        If True, pass ``add_drives_from_params=True`` to the ``calcium_model`` call for
        testing deprecated legacy behavior.
    mesh_shape : tuple of int | None, default=None
        Shape of the cell grid. If None (default), a ``mesh_shape`` of ``(10, 10)`` is
        used. Incompatible with ``reduced=True``.
    reduced : bool, default=False
        If True, use a small ``(3, 3)`` mesh. Incompatible with ``mesh_shape`` of value
        other than None. Does not change the times of drives, unlike the Neymotin 2020
        model fixture.
    electrode_array : dict | None, default=None
        Mapping of electrode array names to their lists of positions, each passed
        to ``net.add_electrode_array``.

    Returns
    -------
    net : Network object
        The Calcium model network.
    inh_name : "basket"
        Name suffix of the inhibitory cell types, which is ``"basket"`` for the
        non-Duecker models.

    Raises
    ------
    ValueError
        If both ``reduced=True`` and ``mesh_shape`` are given.
    """

    def _fix_net_calcium(
        add_drives_from_params=False,
        legacy_mode=False,
        mesh_shape=None,
        reduced=False,
        electrode_array=None,
    ):
        if reduced and mesh_shape:
            raise ValueError(
                "Cannot specify both `reduced=True` and `mesh_shape` argument."
            )
        # Account for Duecker name variations
        inh_name = "basket"

        if reduced:
            mesh_shape = (3, 3)
        elif mesh_shape is not None:
            pass  # Use the provided mesh_shape
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

        return net, inh_name

    return _fix_net_calcium


@pytest.fixture(scope="session")
def fix_net_law_2021():
    """Network fixture for tests of the "Law" model network (``law_2021_model``).

    You can use this and other network fixtures in one of two ways. All network fixtures
    can be used in the following ways:

    1. If you only want to use this network, and do not want to parametrize the test
    across multiple network models, then you can use this like a fixture that accepts
    arguments. For example, in ``test_dipole.py::test_dipole_simulation``, you include
    the fixture in the test's arguments:

    ```
    def test_dipole_simulation(fix_net_neymotin_2020, fix_default_params):
    ```

    and then use the fixture similarly to a regular network models function:

    ```
    net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
    ```

    2. If you want to use this network fixture while parametrizing a test to use other
    network fixtures as well, such as in
    ``test_extracellular.py::test_transmembrane_currents``, then you have to do four
    separate things:

    2.1. Add this fixture as an entry to the ``@pytest.mark.parametrize`` decorator such
    that this network fixture is a value in the list of values given to the "parameter"
    variable. For example:

    ```
    @pytest.mark.parametrize(
        "fix_net_model", ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
    )
    ```

    In the above, ``fix_net_model`` will be a variable that you pass to the argument list
    of the test, and the entries in the list are the values that that parameter will
    take, as the test is re-run once per parameter set.

    2.2. Add both the built-in ``request`` fixture along with the above "parameter"
    variable to your test function's arguments, e.g.

    ```
    def test_transmembrane_currents(fix_net_model, request):
    ```

    2.2. Create the callable network function by requesting the factory fixture's value,
    e.g.
    ```
    net_model = request.getfixturevalue(fix_net_model)
    ```

    (Note that this does NOT create the actual network, it only creates the
    *function/callable* to create the network).

    2.3. Finally, use your new callable to create the network, using the Parameters
    described below, e.g.

    ```
    net, inh_name = net_model(add_drives_from_params=True, reduced=True)
    ```

    This is what actually deploys the appropriate fixture and builds your model, using
    the arguments provided below.

    There are many examples of this pattern in the tests, so if you are unsure how to
    use this, see existing tests.

    Parameters
    ----------
    add_drives_from_params : bool, default=False
        If True, pass ``add_drives_from_params=True`` to the ``law_2021_model`` call in
        order to add the three canonical evoked drives (``evprox1``, ``evdist1``, and
        ``evprox2``).
    legacy_mode : bool, default=False
        If True, pass ``add_drives_from_params=True`` to the ``law_2021_model`` call for
        testing deprecated legacy behavior.
    mesh_shape : tuple of int | None, default=None
        Shape of the cell grid. If None (default), a ``mesh_shape`` of ``(10, 10)`` is
        used. Incompatible with ``reduced=True``.
    reduced : bool, default=False
        If True, use a small ``(3, 3)`` mesh. Incompatible with ``mesh_shape`` of value
        other than None. Does not change the times of drives, unlike the Neymotin 2020
        model fixture.
    electrode_array : dict | None, default=None
        Mapping of electrode array names to their lists of positions, each passed
        to ``net.add_electrode_array``.

    Returns
    -------
    net : Network object
        The Law model network.
    inh_name : "basket"
        Name suffix of the inhibitory cell types, which is ``"basket"`` for the
        non-Duecker models.

    Raises
    ------
    ValueError
        If both ``reduced=True`` and ``mesh_shape`` are given.
    """

    def _fix_net_law_2021(
        add_drives_from_params=False,
        legacy_mode=False,
        mesh_shape=None,
        reduced=False,
        electrode_array=None,
    ):
        if reduced and mesh_shape:
            raise ValueError(
                "Cannot specify both `reduced=True` and `mesh_shape` argument."
            )

        # Account for Duecker name variations
        inh_name = "basket"
        if reduced:
            mesh_shape = (3, 3)
        elif mesh_shape is not None:
            pass  # Use the provided mesh_shape
        else:
            mesh_shape = (10, 10)
        # Legacy mode necessary for exact dipole comparison test
        net = law_2021_model(
            add_drives_from_params=add_drives_from_params,
            legacy_mode=legacy_mode,
            mesh_shape=mesh_shape,
        )
        if electrode_array is not None:
            for name, positions in electrode_array.items():
                net.add_electrode_array(name, positions)

        return net, inh_name

    return _fix_net_law_2021


@pytest.fixture(scope="module")
def fix_run_simulation():
    """Factory fixture that simulates a network and runs basic sanity checks.

    Include ``fix_run_simulation`` in your test's arguments, then call it on a network
    (e.g. one created by a network fixture such as ``fix_net_neymotin_2020``):

    ```
    net, _ = fix_net_neymotin_2020(add_drives_from_params=True, reduced=True)
    dpls, net = fix_run_simulation(net, tstop=40, backend="joblib", n_jobs=2)
    ```

    After simulating, this checks that the network is still picklable and that every
    external drive has one set of events per simulated trial.

    Future refactor: This isn't used very commonly, and may be better just using the raw
    simulate code and backends in-place in the relevant tests, just to make this more
    explicit.

    Parameters
    ----------
    net : Network object
        The network to simulate. It is modified in place by the simulation, and also
        returned by this fixture.
    tstop : float
        The simulation stop time (ms).
    dt : float, default=0.025
        The integration time step (ms).
    n_trials : int, default=2
        The number of trials to simulate. Note that this differs from the default of
        ``simulate_dipole`` in that it defaults to multiple trials!
    record_vsec : 'all' | 'soma' | False, default=False
        Passed to ``simulate_dipole``; which section voltages to record.
    record_isec : 'all' | 'soma' | False, default=False
        Passed to ``simulate_dipole``; which section synaptic currents to record.
    record_ca : 'all' | 'soma' | False, default=False
        Passed to ``simulate_dipole``; which section calcium concentrations to record.
    postproc : bool, default=False
        Passed to ``simulate_dipole`` (deprecated there); whether to apply smoothing
        and scaling to the dipoles.
    verbose : bool, default=True
        Passed to ``simulate_dipole``.
    baseline_correction : bool, default=True
        Passed to ``simulate_dipole``.
    backend : 'mpi' | 'joblib' | None, default=None
        The parallel backend to simulate within. If ``'mpi'``, uses ``MPIBackend``
        with ``n_procs`` and ``mpi_cmd="mpiexec"``. If ``'joblib'``, uses
        ``JoblibBackend`` with ``n_jobs``. If None, no backend context is entered.
    n_procs : int | None, default=None
        The number of MPI processes. Only used when ``backend='mpi'``.
    n_jobs : int, default=1
        The number of Joblib jobs. Only used when ``backend='joblib'``.

    Returns
    -------
    dpls : list of Dipole
        The simulated dipoles, one per trial.
    net : Network object
        The simulated network (the same object passed in).
    """

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
        baseline_correction=True,
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
                    baseline_correction=baseline_correction,
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
                    baseline_correction=baseline_correction,
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
                baseline_correction=baseline_correction,
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


@pytest.fixture(scope="session")
def _base_simulation_cached():
    """Private helper session-cached factory that simulates each network model and variation once.

    This is a private helper fixture; tests should normally use ``fix_use_cached_sims``
    instead of requesting this directly. This is currently only used in `test_viz.py`
    where we're not interested in actually testing that simulation or data output
    themselves are correct, but may be useful for other simulation variants elsewhere in
    the tests for the future.

    The returned callable builds a reduced (``reduced=True``) network from the given
    network fixture, adds two bursty "beta" drives (``beta_prox`` at the proximal
    location and ``beta_dist`` at the distal location), and simulates it with
    ``tstop=100.0``, ``n_trials=2``, and ``record_vsec="all"``. The drives' AMPA
    weights depend on the variation:

    - ``"yes_spikes"``: weights strong enough that the pyramidal cells spike.
    - ``"no_spikes"``: weights so weak that the pyramidal cells do not spike, for
      testing code paths that must handle empty spike data.

    Because simulations are expensive, each ``(net_model_name, variation)`` pair is only
    simulated once per test session, and the result is stored in a cache. Every call
    returns a deep copy of the cached result, so callers may freely modify the returned
    network and dipoles without affecting other tests.

    Parameters
    ----------
    net_model_name : str
        Name of the network fixture, e.g. ``"fix_net_neymotin_2020"``. Used as part
        of the cache key.
    net_model : callable
        The network fixture's callable (e.g. obtained via
        ``request.getfixturevalue(net_model_name)``), which must accept
        ``reduced=True`` and return ``(net, inh_name)``.
    variation : "yes_spikes" | "no_spikes"
        Which set of drive weights to use.

    Returns
    -------
    net : Network object
        A deep copy of the simulated network, including its ``cell_response``.
    dpls : list of Dipole
        A deep copy of the simulated dipoles, one per trial.
    inh_name : str
        Name suffix of the inhibitory cell types for this network model (e.g.
        ``"basket"``).
    """
    cache = {}
    # AMPA weights of the bursty drives for each variation
    variation_weights_ampa = {
        "yes_spikes": {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0},
        "no_spikes": {"L2_pyramidal": 5.4e-5, "L5_pyramidal": 5.4e-5},
    }

    def _get_simulation(net_model_name, net_model, variation):
        key = (net_model_name, variation)
        if key not in cache:
            net, inh_name = net_model(reduced=True)
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
    params=[
        (net_model_name, variation)
        for net_model_name in ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
        for variation in ["yes_spikes", "no_spikes"]
    ],
    ids=lambda param: f"{param[0]}-{param[1]}",
)
def fix_use_cached_sims(_base_simulation_cached, request):
    """Parametrized fixture providing a fresh copy of a cached, already-run simulation.

    This is intended for tests (such as those in ``test_viz.py``) that need a simulated
    network with spiking results but do not care about testing how the simulation was
    set up or executed. See ``_base_simulation_cached`` for the details of the
    simulation.

    By default, this fixture is parametrized over every combination of:

    - network model: ``"fix_net_neymotin_2020"`` and ``"fix_net_duecker_ET"``
    - variation: ``"yes_spikes"`` and ``"no_spikes"``

    so any test that requests it is run once per combination (4 times in total). Each
    combination is only simulated once per test session, and each test receives its
    own deep copy, so tests may modify the returned objects.

    To restrict a test to a subset of combinations, override the parameters using
    "indirect parametrization" (aka ask AI for help), e.g.

    ```
    @pytest.mark.parametrize(
        "fix_use_cached_sims",
        [
            (net_model_name, "yes_spikes")
            for net_model_name in ["fix_net_neymotin_2020", "fix_net_duecker_ET"]
        ],
        ids=lambda param: f"{param[0]}-{param[1]}",
        indirect=True,
    )
    def test_spikes_raster_dipole_overlay(self, fix_use_cached_sims):
        net, dpls, _, _ = fix_use_cached_sims
    ```

    Parameters
    ----------
    This takes no user-provided arguments, and instead requires only other fixtures that
    are automatically provided via the argument list.

    Returns
    -------
    net : Network object
        A copy of the simulated network, including its ``cell_response``.
    dpls : list of Dipole
        A copy of the simulated dipoles, one per trial.
    inh_name : str
        Name suffix of the inhibitory cell types for this network model (e.g.
        ``"basket"``). Useful for building cell type names that work across models.
    variation : "yes_spikes" | "no_spikes"
        Which variation was simulated, so that tests can branch on whether spikes are
        expected.
    """
    net_model_name, variation = request.param
    net_model = request.getfixturevalue(net_model_name)
    net, dpls, inh_name = _base_simulation_cached(
        net_model_name, net_model, variation=variation
    )
    return net, dpls, inh_name, variation


@pytest.fixture
def fix_default_params():
    """Default "flat JSON" parameters for the Neymotin 2020 (aka Jones 2009) model.

    Loads ``hnn_core/param/default.json`` with ``read_params``, freshly for every test,
    so tests are free to modify the returned object.

    Returns
    -------
    params : Params object
        The default parameters, as read from ``default.json``.
    """
    params_fname = hnn_core_root / "param" / "default.json"
    return read_params(params_fname)
