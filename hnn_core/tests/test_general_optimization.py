# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import os.path as op

import pytest

import hnn_core
from hnn_core import jones_2009_model, simulate_dipole, read_params
from hnn_core.optimization import Optimizer


@pytest.mark.parametrize("solver", ['bayesian', 'cobyla'])
def test_optimize_evoked(solver):
    """Test optimization routines for evoked drives in a reduced network."""

    max_iter = 11
    tstop = 10.
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3})
    net_orig = jones_2009_model(params)

    mu_orig = 2.
    weights_ampa = {'L2_basket': 0.5,
                    'L2_pyramidal': 0.5,
                    'L5_basket': 0.5,
                    'L5_pyramidal': 0.5}
    synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                       'L5_basket': 1., 'L5_pyramidal': 1.}
    net_orig.add_evoked_drive('evprox',
                              mu=mu_orig,
                              sigma=1,
                              numspikes=1,
                              location='proximal',
                              weights_ampa=weights_ampa,
                              synaptic_delays=synaptic_delays)
    dpl_orig = simulate_dipole(net_orig, tstop=tstop, n_trials=n_trials)[0]

    # define set_params function and constraints
    net_offset = jones_2009_model(params)

    def set_params(net_offset, params):
        weights_ampa = {'L2_basket': 0.5,
                        'L2_pyramidal': 0.5,
                        'L5_basket': 0.5,
                        'L5_pyramidal': 0.5}
        synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                           'L5_basket': 1., 'L5_pyramidal': 1.}
        net_offset.add_evoked_drive('evprox',
                                    mu=params['mu'],
                                    sigma=params['sigma'],
                                    numspikes=1,
                                    location='proximal',
                                    weights_ampa=weights_ampa,
                                    synaptic_delays=synaptic_delays)

    # define constraints
    constraints = dict()
    constraints.update({'mu': (1, 6),
                        'sigma': (1, 3)})

    optim = Optimizer(net_offset, tstop=tstop, constraints=constraints,
                      set_params=set_params, solver=solver,
                      obj_fun='dipole_rmse', max_iter=max_iter)

    # test exception raised
    with pytest.raises(ValueError, match='The current Network instance has '
                       'external drives, provide a Network object with no '
                       'external drives.'):
        net_with_drives = net_orig.copy()
        optim = Optimizer(net_with_drives,
                          tstop=tstop,
                          constraints=constraints,
                          set_params=set_params,
                          solver=solver,
                          obj_fun='dipole_rmse',
                          max_iter=max_iter)

    # test repr before fitting
    assert 'fit=False' in repr(optim), "optimizer is already fit"

    optim.fit(dpl_orig)

    # test repr after fitting
    assert 'fit=True' in repr(optim), "optimizer was not fit"

    # the optimized parameter is in the range
    for param_idx, param in enumerate(optim.opt_params_):
        assert list(constraints.values())[param_idx][0] \
            <= param \
            <= list(constraints.values())[param_idx][1], \
            "Optimized parameter is not in user-defined range"

    obj = optim.obj_
    # the number of returned rmse values should be the same as max_iter
    assert len(obj) <= max_iter, \
           "Number of rmse values should be the same as max_iter"
    # the returned rmse values should be positive
    assert all(vals >= 0 for vals in obj), "rmse values should be positive"
