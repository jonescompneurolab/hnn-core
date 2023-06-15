from hnn_core import simulate_dipole
import numpy as np


def _rmse_evoked(net, param_names, set_params, predicted_params, scaling,
                 target_statistic, f_bands, weights):
    """The objective function for evoked responses.

       Parameters
       -----------
       net : Network
       param_names : dictionary
           Parameters to change.
       target_statistic : ndarray
           The recorded dipole.
       predicted_params : list
           Parameters selected by the optimizer.

       Returns
       -------
       rmse : normalized RMSE between recorded and simulated dipole
    """

    # get network with predicted params
    new_net = set_params(net, param_names, predicted_params) # ???
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop=500, n_trials=1)[0]

    # smooth & scale
    dpl.scale(scaling)
    dpl.smooth(20)

    # calculate rmse
    obj = np.sqrt(((dpl.data['agg'] - target_statistic)**2).sum() / len(dpl.times)) / (max(target_statistic) - min(target_statistic))
    return obj


def _rmse_rhythmic(net, param_names, set_params, predicted_params, scaling,
                   target_statistic, f_bands, weights):
    """The objective function for evoked responses.

       Parameters
       -----------
       net : Network
       param_names : dictionary
           Parameters to change.
       target_statistic : ndarray
           Recorded dipole.
       predicted_params : list
           Parameters selected by the optimizer.
       compute_psd :
           ...

       Returns
       -------
       rmse : normalized RMSE between recorded and simulated dipole
    """

    from scipy.signal import periodogram

    # simulate dpl with predicted params
    new_net = set_params(net, param_names, predicted_params) # ???
    dpl = simulate_dipole(new_net, tstop=300, n_trials=1)[0]

    # scale
    dpl.scale(scaling)

    # get psd of simulated dpl
    freqs_simulated, psd_simulated = periodogram(dpl.data['agg'], dpl.sfreq,
                                                 window='hamming')

    # for each f band
    f_bands_psds = list()
    for f_band_idx, f_band_val in enumerate(f_bands):
        f_band_psd_idx = np.where(np.logical_and(freqs_simulated >= f_band_val[0],
                                                 freqs_simulated <= f_band_val[1]))[0]
        f_bands_psds.append((-weights[f_band_idx] * sum(psd_simulated[f_band_psd_idx])) / sum(psd_simulated))

    # grand sum
    obj = sum(f_bands_psds)

    return obj


def _rmse_poisson():
    return