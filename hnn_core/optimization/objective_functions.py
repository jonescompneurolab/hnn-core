"""Objective functions for parameter optimization."""

# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from hnn_core import simulate_dipole
from ..dipole import _rmse


def _rmse_evoked(initial_net, initial_params, set_params, predicted_params,
                 update_params, obj_values, tstop, obj_fun_kwargs):
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
    tstop : float
        The simulated dipole's duration.
    target : instance of Dipole
        A dipole object with experimental data.

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
    if 'scale_factor' in obj_fun_kwargs:
        dpl.scale(obj_fun_kwargs['scale_factor'])
    if 'smooth_window_len' in obj_fun_kwargs:
        dpl.smooth(obj_fun_kwargs['smooth_window_len'])

    obj = _rmse(dpl, obj_fun_kwargs['target'], tstop=tstop)

    obj_values.append(obj)

    return obj


def _maximize_psd(initial_net, initial_params, set_params, predicted_params,
                  update_params, obj_values, tstop, obj_fun_kwargs):
    """The objective function for PSDs.

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
    tstop : float
        The simulated dipole's duration.
    f_bands : list of tuples
        Lower and higher limit for each frequency band.
    relative_bandpower : tuple
        Weight for each frequency band.

    Returns
    -------
    obj : float
        Sum of the weighted frequency band PSDs relative to total signal PSD.

    Notes
    -----
    The objective function minimizes the sum of the weighted (user-defined)
    frequency band PSDs (user-defined) relative to the total PSD of the signal.
    The objective function can be represented as -Σc[ΣPSD(i)/ΣPSD(j)] where c
    is the weight for each frequency band, PSD(i) is the PSD for each frequency
    band, and PSD(j) is the total PSD of the signal.
    """

    import numpy as np

    from scipy.signal import periodogram

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)
    dpl = simulate_dipole(new_net, tstop=tstop, n_trials=1)[0]

    # smooth & scale
    if 'scale_factor' in obj_fun_kwargs:
        dpl.scale(obj_fun_kwargs['scale_factor'])
    if 'smooth_window_len' in obj_fun_kwargs:
        dpl.smooth(obj_fun_kwargs['smooth_window_len'])

    # resample?

    # get psd of simulated dpl
    freqs_simulated, psd_simulated = periodogram(dpl.data['agg'], dpl.sfreq,
                                                 window='hamming')

    # for each f band
    f_bands_psds = list()
    for idx, f_band in enumerate(obj_fun_kwargs['f_bands']):
        f_band_idx = np.where(np.logical_and(freqs_simulated >= f_band[0],
                                             freqs_simulated <= f_band[1]))[0]
        f_bands_psds.append(-obj_fun_kwargs['relative_bandpower'][idx] *
                            sum(psd_simulated[f_band_idx]))

    # grand sum
    obj = sum(f_bands_psds) / sum(psd_simulated)

    obj_values.append(obj)

    return obj
