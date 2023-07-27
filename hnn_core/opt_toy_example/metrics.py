import numpy as np

from hnn_core import simulate_dipole, MPIBackend

from scipy.signal import resample


def _rmse_evoked(net, initial_params, set_params, predicted_params,
                 update_params, obj_values, scale_factor, smooth_window_len,
                 tstop, target_statistic):
    """
    The objective function for evoked responses.

    Parameters
    ----------
    net : Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update param_dict.
    scale_factor : float
        The dipole scale factor.
    smooth_window_len : float
        The smooth window length.
    tstop : float
        The simulated dipole's duration.
    target_statistic : ndarray
        The recorded dipole.

    Returns
    -------
    obj : float
        Normalized RMSE between recorded and simulated dipole.
    """

    param_dict = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = set_params(net.copy(), param_dict)
    with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
        dpl = simulate_dipole(new_net, tstop=tstop, n_trials=1)[0]

    # downsample if necessary
    if (len(dpl.data['agg']) < len(target_statistic)):
        target_statistic = resample(target_statistic, len(dpl.data['agg']))
    elif (len(dpl.data['agg']) > len(target_statistic)):
        dpl.data['agg'] = resample(dpl.data['agg'], len(target_statistic))

    # smooth & scale
    dpl.scale(scale_factor)
    dpl.smooth(smooth_window_len)

    # calculate rmse
    obj = np.sqrt(((dpl.data['agg'] - target_statistic)**2).sum() /
                  len(dpl.times)) / (max(target_statistic) -
                                     min(target_statistic))

    obj_values.append(obj)

    return obj
