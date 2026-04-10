# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import pytest

from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.optimization import Optimizer
import numpy as np


@pytest.mark.parametrize("solver", ["bayesian", "cobyla", "cma"])
@pytest.mark.parametrize("obj_fun", ["dipole_rmse", "dipole_corr"])
def test_optimize_evoked(solver, obj_fun):
    """Test optimization routines for evoked drives in a reduced network."""

    max_iter = 2
    tstop = 10.0
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    net_orig = jones_2009_model(mesh_shape=(3, 3))

    mu_orig = 2.0
    weights_ampa = {
        "L2_basket": 0.5,
        "L2_pyramidal": 0.5,
        "L5_basket": 0.5,
        "L5_pyramidal": 0.5,
    }
    synaptic_delays = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    net_orig.add_evoked_drive(
        "evprox",
        mu=mu_orig,
        sigma=1,
        numspikes=1,
        location="proximal",
        weights_ampa=weights_ampa,
        synaptic_delays=synaptic_delays,
    )
    dpl_orig = simulate_dipole(net_orig, tstop=tstop, n_trials=n_trials)[0]

    # define set_params function and constraints
    net_offset = jones_2009_model(mesh_shape=(3, 3))

    def set_params(net_offset, params):
        weights_ampa = {
            "L2_basket": 0.5,
            "L2_pyramidal": 0.5,
            "L5_basket": 0.5,
            "L5_pyramidal": 0.5,
        }
        synaptic_delays = {
            "L2_basket": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 1.0,
            "L5_pyramidal": 1.0,
        }
        net_offset.add_evoked_drive(
            "evprox",
            mu=params["mu"],
            sigma=params["sigma"],
            numspikes=1,
            location="proximal",
            weights_ampa=weights_ampa,
            synaptic_delays=synaptic_delays,
        )

    # define constraints
    constraints = dict()
    constraints.update({"mu": (1, 6), "sigma": (1, 3)})

    optim = Optimizer(
        net_offset,
        tstop=tstop,
        constraints=constraints,
        set_params=set_params,
        solver=solver,
        obj_fun=obj_fun,
        max_iter=max_iter,
    )

    # test repr before fitting
    assert "fit=False" in repr(optim), "optimizer is already fit"

    optim.fit(target=dpl_orig, n_trials=3, scale_factor=3000, smooth_window_len=1)

    # test repr after fitting
    assert "fit=True" in repr(optim), "optimizer was not fit"

    # the optimized parameter is in the range
    for param_idx, param in enumerate(optim.opt_params_):
        assert (
            list(constraints.values())[param_idx][0]
            <= param
            <= list(constraints.values())[param_idx][1]
        ), "Optimized parameter is not in user-defined range"

    obj = optim.obj_
    # the number of returned loss values should be the same as max_iter
    assert len(obj) <= max_iter, "Number of loss values should be the same as max_iter"
    # the returned loss values should be positive
    assert all(vals >= 0 for vals in obj), "loss values should be positive"


@pytest.mark.parametrize("solver", ["bayesian", "cobyla", "cma"])
@pytest.mark.parametrize("relative_bandpower", [[1, 2], 0.5])
def test_rhythmic(solver, relative_bandpower):
    """Test optimization routines for rhythmic drives in a reduced network."""

    max_iter = 2
    tstop = 10.0

    # simulate a dipole to establish ground-truth drive parameters
    net_offset = jones_2009_model(mesh_shape=(3, 3))

    # define set_params function and constraints
    def set_params(net_offset, params):
        # Proximal (alpha)
        weights_ampa_p = {
            "L2_pyramidal": params["alpha_prox_weight"],
            "L5_pyramidal": 4.4e-5,
        }
        syn_delays_p = {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0}

        net_offset.add_bursty_drive(
            "alpha_prox",
            tstart=params["alpha_prox_tstart"],
            burst_rate=params["alpha_prox_burst_rate"],
            burst_std=params["alpha_prox_burst_std"],
            numspikes=2,
            spike_isi=10,
            n_drive_cells=10,
            location="proximal",
            weights_ampa=weights_ampa_p,
            synaptic_delays=syn_delays_p,
        )

        # Distal (beta)
        weights_ampa_d = {
            "L2_pyramidal": params["alpha_dist_weight"],
            "L5_pyramidal": 4.4e-5,
        }
        syn_delays_d = {"L2_pyramidal": 5.0, "L5_pyramidal": 5.0}

        net_offset.add_bursty_drive(
            "alpha_dist",
            tstart=params["alpha_dist_tstart"],
            burst_rate=params["alpha_dist_burst_rate"],
            burst_std=params["alpha_dist_burst_std"],
            numspikes=2,
            spike_isi=10,
            n_drive_cells=10,
            location="distal",
            weights_ampa=weights_ampa_d,
            synaptic_delays=syn_delays_d,
        )

    # define constraints
    constraints = dict()
    constraints.update(
        {
            "alpha_prox_weight": (4.4e-5, 6.4e-5),
            "alpha_prox_tstart": (45, 55),
            "alpha_prox_burst_rate": (8, 12),
            "alpha_prox_burst_std": (10, 25),
            "alpha_dist_weight": (4.4e-5, 6.4e-5),
            "alpha_dist_tstart": (45, 55),
            "alpha_dist_burst_rate": (8, 12),
            "alpha_dist_burst_std": (10, 25),
        }
    )

    # Optimize
    optim = Optimizer(
        net_offset,
        tstop=tstop,
        constraints=constraints,
        set_params=set_params,
        solver=solver,
        obj_fun="maximize_psd",
        max_iter=max_iter,
    )

    # test repr before fitting
    assert "fit=False" in repr(optim), "optimizer is already fit"

    with pytest.raises(ValueError, match="Length of relative_bandpower"):
        optim.fit(f_bands=[(8, 12), (18, 22)], relative_bandpower=[1, 2, 3])

    optim.fit(f_bands=[(8, 12), (18, 22)], relative_bandpower=relative_bandpower)

    # test repr after fitting
    assert "fit=True" in repr(optim), "optimizer was not fit"

    # the optimized parameter is in the range
    for param_idx, param in enumerate(optim.opt_params_):
        assert (
            list(constraints.values())[param_idx][0]
            <= param
            <= list(constraints.values())[param_idx][1]
        ), "Optimized parameter is not in user-defined range"

    obj = optim.obj_
    # the number of returned rmse values should be the same as max_iter
    assert len(obj) <= max_iter, "Number of rmse values should be the same as max_iter"


@pytest.mark.parametrize("solver", ["bayesian", "cobyla", "cma"])
def test_initial_params(solver):
    """Test optimization routines with user-defined initial parameters."""

    max_iter = 2
    tstop = 10.0
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    net_orig = jones_2009_model(mesh_shape=(3, 3))

    mu_orig = 2.0
    sigma_orig = 1.0
    weights_ampa = {
        "L2_basket": 0.5,
        "L2_pyramidal": 0.5,
        "L5_basket": 0.5,
        "L5_pyramidal": 0.5,
    }
    synaptic_delays = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    net_orig.add_evoked_drive(
        "evprox",
        mu=mu_orig,
        sigma=sigma_orig,
        numspikes=1,
        location="proximal",
        weights_ampa=weights_ampa,
        synaptic_delays=synaptic_delays,
    )
    dpl_orig = simulate_dipole(net_orig, tstop=tstop, n_trials=n_trials)[0]

    # define set_params function and constraints
    net_offset = jones_2009_model(mesh_shape=(3, 3))

    def set_params(net_offset, params):
        weights_ampa = {
            "L2_basket": 0.5,
            "L2_pyramidal": 0.5,
            "L5_basket": 0.5,
            "L5_pyramidal": 0.5,
        }
        synaptic_delays = {
            "L2_basket": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 1.0,
            "L5_pyramidal": 1.0,
        }
        net_offset.add_evoked_drive(
            "evprox",
            mu=params["mu"],
            sigma=params["sigma"],
            numspikes=1,
            location="proximal",
            weights_ampa=weights_ampa,
            synaptic_delays=synaptic_delays,
        )

    # define constraints
    constraints = dict()
    constraints.update({"mu": (1, 10), "sigma": (1, 10)})
    initial_params = {"mu": 5, "sigma": 5}

    optim = Optimizer(
        net_offset,
        tstop=tstop,
        constraints=constraints,
        set_params=set_params,
        solver=solver,
        obj_fun="dipole_rmse",
        max_iter=max_iter,
        initial_params=initial_params,
    )

    optim.fit(target=dpl_orig, n_trials=3)

    # Test that the initial_params were correctly set by the user
    assert optim.initial_params == initial_params
    assert optim.initial_params["mu"] == 5
    assert optim.initial_params["sigma"] == 5

    optim.fit(target=dpl_orig, n_trials=3)


@pytest.mark.parametrize("solver", ["bayesian", "cobyla", "cma"])
@pytest.mark.parametrize(
    "initial_params, error_type",
    [
        # initial_params is not a dict
        ([5, 5], TypeError),
        # initial_params keys do not match constraints keys
        ({"mu": 5, "wrong_key": 5}, ValueError),
        # initial_params values are not float or int
        ({"mu": "five", "sigma": 5}, TypeError),
        # initial_params values are outside the range of constraints
        ({"mu": 11, "sigma": 5}, ValueError),
    ],
)
def test_initial_params_validation(solver, initial_params, error_type):
    """Test initial_params validation."""

    tstop = 10.0
    net_offset = jones_2009_model(mesh_shape=(3, 3))

    def set_params(net_offset, params):
        weights_ampa = {
            "L2_basket": 0.5,
            "L2_pyramidal": 0.5,
            "L5_basket": 0.5,
            "L5_pyramidal": 0.5,
        }
        synaptic_delays = {
            "L2_basket": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 1.0,
            "L5_pyramidal": 1.0,
        }
        net_offset.add_evoked_drive(
            "evprox",
            mu=params["mu"],
            sigma=params["sigma"],
            numspikes=1,
            location="proximal",
            weights_ampa=weights_ampa,
            synaptic_delays=synaptic_delays,
        )

    # define constraints
    constraints = dict()
    constraints.update({"mu": (1, 10), "sigma": (1, 10)})

    with pytest.raises(error_type):
        Optimizer(
            net_offset,
            tstop=tstop,
            constraints=constraints,
            set_params=set_params,
            solver=solver,
            obj_fun="dipole_rmse",
            max_iter=2,
            initial_params=initial_params,
        )


def test_cma_validation():
    net = jones_2009_model(mesh_shape=(3, 3))
    tstop = 10.0
    constraints = {"mu": (1, 10), "sigma": (1, 10)}
    solver = "cma"
    obj_fun = "dipole_rmse"
    max_iter = 2

    dpl_target = simulate_dipole(net, tstop=tstop)[0]

    def set_params(a, b):
        pass

    optim = Optimizer(
        net,
        tstop=tstop,
        constraints=constraints,
        set_params=set_params,
        solver=solver,
        obj_fun=obj_fun,
        max_iter=max_iter,
    )

    with pytest.raises(ValueError, match="sigma0 must be greater than"):
        optim.fit(target=dpl_target, sigma0=-1)

    with pytest.raises(ValueError, match="it must be shape"):
        optim.fit(target=dpl_target, sigma0=np.array([[0, 1], [2, 3]]))

    with pytest.raises(ValueError, match="length must be the same as the constraints"):
        optim.fit(target=dpl_target, sigma0=[1, 2, 3])

    optim.fit(target=dpl_target, sigma0=[1, 2])
