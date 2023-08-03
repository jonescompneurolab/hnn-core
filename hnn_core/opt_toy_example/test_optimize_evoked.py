# Authors: Mainak Jas <mainakjas@gmail.com>
#          Carolina Fernandez <cxf418@miami.edu>

from hnn_core import jones_2009_model, simulate_dipole
from general import Optimizer  # change path***


def _optimize_evoked(solver):
    """Test running the full routine in a reduced network."""

    tstop = 5.
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    net_orig = jones_2009_model()
    mu_orig = 6.
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

    def set_params(net_offset, param_dict):
        weights_ampa = {'L2_basket': 0.5,
                        'L2_pyramidal': 0.5,
                        'L5_basket': 0.5,
                        'L5_pyramidal': 0.5}
        synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                           'L5_basket': 1., 'L5_pyramidal': 1.}
        net_offset.add_evoked_drive('evprox',
                                    mu=param_dict['mu_offset'],
                                    sigma=1,
                                    numspikes=1,
                                    location='proximal',
                                    weights_ampa=weights_ampa,
                                    synaptic_delays=synaptic_delays)

    # define constraints
    mu_offset = 4.  # initial time-shifted drive
    mu_range = (2, 8)
    constraints = dict()
    constraints.update({'mu_offset': mu_range})

    optim = Optimizer(net_offset, constraints=constraints,
                      set_params=set_params, solver=solver,
                      obj_fun='evoked', tstop=tstop)
    optim.fit(dpl_orig.data['agg'])

    opt_param = optim.opt_params
    # the optimized parameter is in the range
    assert opt_param[0] in range(mu_range[0], mu_range[1]), "Optimized parameter is not in user-defined range"

    obj = optim.obj
    # the number of returned rmse values should be the same as max_iter
    assert len(obj) <= 200, "Number of rmse values should be the same as max_iter"
    # the returned rmse values should be positive
    assert all(vals > 0 for vals in obj), "rmse values should be positive"


def test_bayesian_evoked():
    _optimize_evoked('bayesian')


def test_cobyla_evoked():
    _optimize_evoked('cobyla')
