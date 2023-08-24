"""Objective functions for parameter optimization."""

# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np

from hnn_core import simulate_dipole

from scipy.signal import resample


def _rmse_evoked(initial_net, initial_params, set_params, predicted_params,
                 update_params, obj_values, scale_factor, smooth_window_len,
                 target, tstop):
    """The objective function for evoked responses.

    Parameters
    ----------
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    scale_factor : float
        The dipole scale factor.
    smooth_window_len : float
        The smooth window length.
    tstop : float
        The simulated dipole's duration.
    target : ndarray
        The recorded dipole.

    Returns
    -------
    obj : float
        Normalized RMSE between recorded and simulated dipole.
    """

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)
    dpl = simulate_dipole(new_net, tstop=tstop, n_trials=1)[0]

    # smooth & scale
    dpl.scale(scale_factor)
    if smooth_window_len is not None:
        dpl.smooth(smooth_window_len)

    # downsample if necessary
    if (len(dpl.data['agg']) < len(target)):
        target = resample(target, len(dpl.data['agg']))
    elif (len(dpl.data['agg']) > len(target)):
        dpl.data['agg'] = resample(dpl.data['agg'], len(target))

    # calculate rmse
    obj = np.sqrt(((dpl.data['agg'] - target)**2).sum() /
                  len(dpl.times)) / (max(target) -
                                     min(target))

    obj_values.append(obj)

    return obj
