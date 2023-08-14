# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from hnn_core import jones_2009_model, simulate_dipole

from hnn_core.optimization import Optimizer

import pytest


@pytest.mark.parametrize("solver", ['bayesian', 'cobyla'])
def test_optimize_evoked(solver):
    """Test optimization routines for evoked drives in a reduced network."""

    tstop = 10.
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    net_orig = jones_2009_model()
    net_orig._N_pyr_x = 3
    net_orig._N_pyr_y = 3
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
    net_offset = jones_2009_model()
    net_offset._N_pyr_x = 3
    net_offset._N_pyr_y = 3

    def set_params(net_offset, params):
        weights_ampa = {'L2_basket': 0.5,
                        'L2_pyramidal': 0.5,
                        'L5_basket': 0.5,
                        'L5_pyramidal': 0.5}
        synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                           'L5_basket': 1., 'L5_pyramidal': 1.}
        net_offset.add_evoked_drive('evprox',
                                    mu=params['mu'],
                                    sigma=1,
                                    numspikes=1,
                                    location='proximal',
                                    weights_ampa=weights_ampa,
                                    synaptic_delays=synaptic_delays)

    # define constraints
    mu_range = (1, 6)
    constraints = dict()
    constraints.update({'mu': mu_range})

    optim = Optimizer(net_offset, tstop=tstop, constraints=constraints,
                      set_params=set_params, solver=solver,
                      obj_fun='dipole_rmse')

    # test repr before fitting
    assert 'fit=False' in repr(optim), "optimizer is already fit"

    optim.fit(dpl_orig.data['agg'])

    # test repr after fitting
    assert 'fit=True' in repr(optim), "optimizer was not fit"

    opt_param = optim.opt_params_[0]
    # the optimized parameter is in the range
    assert mu_range[0] <= opt_param <= mu_range[1], \
        "Optimized parameter is not in user-defined range"

    obj = optim.obj_
    # the number of returned rmse values should be the same as max_iter
    assert len(obj) <= 200, \
        "Number of rmse values should be the same as max_iter"
    # the returned rmse values should be positive
    assert all(vals >= 0 for vals in obj), "rmse values should be positive"
