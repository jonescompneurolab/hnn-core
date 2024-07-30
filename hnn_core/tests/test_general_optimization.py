# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import pytest

from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.optimization import Optimizer


@pytest.mark.parametrize("solver", ['bayesian', 'cobyla'])
def test_optimize_evoked(solver):
    """Test optimization routines for evoked drives in a reduced network."""

    max_iter = 2
    tstop = 10.
    n_trials = 1

    # simulate a dipole to establish ground-truth drive parameters
    net_orig = jones_2009_model(mesh_shape=(3, 3))

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
    net_offset = jones_2009_model(mesh_shape=(3, 3))

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

    optim.fit(target=dpl_orig)

    # test repr after fitting
    assert 'fit=True' in repr(optim), "optimizer was not fit"

    # the optimized parameter is in the range
    for param_idx, param in enumerate(optim.opt_params_):
        assert (list(constraints.values())[param_idx][0] <= param <=
                list(constraints.values())[param_idx][1]), (
                    "Optimized parameter is not in user-defined range")

    obj = optim.obj_
    # the number of returned rmse values should be the same as max_iter
    assert (len(obj) <= max_iter), (
           "Number of rmse values should be the same as max_iter")
    # the returned rmse values should be positive
    assert all(vals >= 0 for vals in obj), "rmse values should be positive"


@pytest.mark.parametrize("solver", ['bayesian', 'cobyla'])
def test_rhythmic(solver):
    """Test optimization routines for rhythmic drives in a reduced network."""

    max_iter = 2
    tstop = 10.

    # simulate a dipole to establish ground-truth drive parameters
    net_offset = jones_2009_model(mesh_shape=(3, 3))

    # define set_params function and constraints
    def set_params(net_offset, params):

        # Proximal (alpha)
        weights_ampa_p = {'L2_pyramidal': params['alpha_prox_weight'],
                          'L5_pyramidal': 4.4e-5}
        syn_delays_p = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}

        net_offset.add_bursty_drive('alpha_prox',
                                    tstart=params['alpha_prox_tstart'],
                                    burst_rate=params['alpha_prox_burst_rate'],
                                    burst_std=params['alpha_prox_burst_std'],
                                    numspikes=2,
                                    spike_isi=10,
                                    n_drive_cells=10,
                                    location='proximal',
                                    weights_ampa=weights_ampa_p,
                                    synaptic_delays=syn_delays_p)

        # Distal (beta)
        weights_ampa_d = {'L2_pyramidal': params['alpha_dist_weight'],
                          'L5_pyramidal': 4.4e-5}
        syn_delays_d = {'L2_pyramidal': 5., 'L5_pyramidal': 5.}

        net_offset.add_bursty_drive('alpha_dist',
                                    tstart=params['alpha_dist_tstart'],
                                    burst_rate=params['alpha_dist_burst_rate'],
                                    burst_std=params['alpha_dist_burst_std'],
                                    numspikes=2,
                                    spike_isi=10,
                                    n_drive_cells=10,
                                    location='distal',
                                    weights_ampa=weights_ampa_d,
                                    synaptic_delays=syn_delays_d)

    # define constraints
    constraints = dict()
    constraints.update({'alpha_prox_weight': (4.4e-5, 6.4e-5),
                        'alpha_prox_tstart': (45, 55),
                        'alpha_prox_burst_rate': (8, 12),
                        'alpha_prox_burst_std': (10, 25),
                        'alpha_dist_weight': (4.4e-5, 6.4e-5),
                        'alpha_dist_tstart': (45, 55),
                        'alpha_dist_burst_rate': (8, 12),
                        'alpha_dist_burst_std': (10, 25)})

    # Optimize
    optim = Optimizer(net_offset, tstop=tstop, constraints=constraints,
                      set_params=set_params, solver=solver,
                      obj_fun='maximize_psd', max_iter=max_iter)

    # test exception raised
    with pytest.raises(ValueError, match='The current Network instance has '
                       'external drives, provide a Network object with no '
                       'external drives.'):
        net_with_drives = jones_2009_model(add_drives_from_params=True)
        optim = Optimizer(net_with_drives,
                          tstop=tstop,
                          constraints=constraints,
                          set_params=set_params,
                          solver=solver,
                          obj_fun='maximize_psd',
                          max_iter=max_iter)

    # test repr before fitting
    assert 'fit=False' in repr(optim), "optimizer is already fit"

    optim.fit(f_bands=[(8, 12), (18, 22)], relative_bandpower=(1, 2))

    # test repr after fitting
    assert 'fit=True' in repr(optim), "optimizer was not fit"

    # the optimized parameter is in the range
    for param_idx, param in enumerate(optim.opt_params_):
        assert (list(constraints.values())[param_idx][0] <= param <=
                list(constraints.values())[param_idx][1]), (
                    "Optimized parameter is not in user-defined range")

    obj = optim.obj_
    # the number of returned rmse values should be the same as max_iter
    assert (len(obj) <= max_iter), (
           "Number of rmse values should be the same as max_iter")


@pytest.mark.parametrize("solver", ['bayesian', 'cobyla'])
def test_user_obj_fun(solver):
    """Test optimization routines with a user-defined optimization function."""

    max_iter = 2
    tstop = 10.

    # simulate a dipole to establish ground-truth drive parameters
    net_offset = jones_2009_model(mesh_shape=(3, 3))

    def maximize_csd(initial_net, initial_params, set_params, predicted_params,
                     update_params, obj_values, tstop, obj_fun_kwargs):

        import numpy as np
        from hnn_core.optimization import _update_params
        from hnn_core.extracellular import (calculate_csd2d,
                                            _get_laminar_z_coords)

        params = _update_params(initial_params, predicted_params)

        # simulate dpl with predicted params
        new_net = initial_net.copy()
        set_params(new_net, params)

        # set electrode array
        depths = list(range(-325, 2150, 100))
        electrode_pos = [(135, 135, dep) for dep in depths]
        new_net.add_electrode_array('shank1', electrode_pos)

        simulate_dipole(new_net, tstop=tstop, dt=0.5, n_trials=1)[0]

        potentials = new_net.rec_arrays['shank1'][0]

        # smooth
        if 'smooth_window_len' in obj_fun_kwargs:
            potentials.smooth(window_len=obj_fun_kwargs['smooth_window_len'])

        # get csd of simulated potentials
        lfp = potentials.voltages[0]  # n_contacts, n_times
        contact_labels, delta = _get_laminar_z_coords(potentials.positions)
        csd = calculate_csd2d(lfp_data=lfp, delta=delta)  # n_contacts, n_times

        # for each tuple
        csd_subsets = list()  # band, n_contacts, n_times
        for idx, t_band in enumerate(obj_fun_kwargs['t_bands']):
            t_min = np.argmax(potentials.times >= t_band[0])
            t_max = np.argmax(potentials.times >= t_band[1])
            depth_min = np.argmax(contact_labels >=
                                  obj_fun_kwargs['electrode_depths'][idx][0])
            depth_max = np.argmax(contact_labels >=
                                  obj_fun_kwargs['electrode_depths'][idx][1])

            csd_subsets.append(sum(sum(csd[depth_min:depth_max + 1,
                                           t_min:t_max + 1])))

        obj = sum(csd_subsets) / sum(sum(csd))
        obj_values.append(obj)

        return obj

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
    constraints.update({'mu': (1, 200),
                        'sigma': (1, 15)})

    optim = Optimizer(net_offset, tstop=tstop, constraints=constraints,
                      set_params=set_params, solver=solver,
                      obj_fun=maximize_csd, max_iter=max_iter)

    # test exception raised
    with pytest.raises(ValueError, match='The current Network instance has '
                       'external drives, provide a Network object with no '
                       'external drives.'):
        net_with_drives = jones_2009_model(add_drives_from_params=True)
        optim = Optimizer(net_with_drives,
                          tstop=tstop,
                          constraints=constraints,
                          set_params=set_params,
                          solver=solver,
                          obj_fun=maximize_csd,
                          max_iter=max_iter)

    # test repr before fitting
    assert 'fit=False' in repr(optim), "optimizer is already fit"

    # increase power in infragranular layers (100-150 ms)
    optim.fit(t_bands=[(100, 150),], electrode_depths=[(0, 200),])

    # test repr after fitting
    assert 'fit=True' in repr(optim), "optimizer was not fit"

    # the optimized parameter is in the range
    for param_idx, param in enumerate(optim.opt_params_):
        assert (list(constraints.values())[param_idx][0] <= param <=
                list(constraints.values())[param_idx][1]), (
                    "Optimized parameter is not in user-defined range")
