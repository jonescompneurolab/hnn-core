"""Objective functions for parameter optimization."""

# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from hnn_core import simulate_dipole
from ..dipole import _rmse, average_dipoles


def _rmse_evoked(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
):
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
    n_trials : int
        Number of trials to simulate and average.

    Returns
    -------
    obj : float
        Normalized RMSE between recorded and simulated dipole.
    """

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)

    dpls = simulate_dipole(new_net, tstop=tstop, n_trials=obj_fun_kwargs["n_trials"])

    # smooth & scale
    if "scale_factor" in obj_fun_kwargs:
        [dpl.scale(obj_fun_kwargs["scale_factor"]) for dpl in dpls]
    if "smooth_window_len" in obj_fun_kwargs:
        [dpl.smooth(obj_fun_kwargs["smooth_window_len"]) for dpl in dpls]

    dpl = average_dipoles(dpls)
    obj = _rmse(dpl, obj_fun_kwargs["target"], tstop=tstop)
    obj_values.append(obj)

    return obj


def _maximize_psd(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
):
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
    relative_bandpower : list of float | float
        Weight for each frequency band in f_bands. If a single float is provided,
        the same weight is applied to all frequency bands.

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
    if "scale_factor" in obj_fun_kwargs:
        dpl.scale(obj_fun_kwargs["scale_factor"])
    if "smooth_window_len" in obj_fun_kwargs:
        dpl.smooth(obj_fun_kwargs["smooth_window_len"])

    # get psd of simulated dpl
    freqs_simulated, psd_simulated = periodogram(
        dpl.data["agg"], dpl.sfreq, window="hamming"
    )

    # for each f band
    f_bands_psds = list()
    relative_bandpower = obj_fun_kwargs["relative_bandpower"]

    # Handle float and list inputs for relative_bandpower
    if isinstance(relative_bandpower, float):
        relative_bandpower = [relative_bandpower] * len(obj_fun_kwargs["f_bands"])
    elif len(relative_bandpower) != len(obj_fun_kwargs["f_bands"]):
        raise ValueError("Length of relative_bandpower must match length of f_bands.")

    for idx, f_band in enumerate(obj_fun_kwargs["f_bands"]):
        f_band_idx = np.where(
            np.logical_and(freqs_simulated >= f_band[0], freqs_simulated <= f_band[1])
        )[0]
        f_bands_psds.append(relative_bandpower[idx] * sum(psd_simulated[f_band_idx]))

    # The optimizer is designed to minimize the objective function.
    # Maximizing the relative band power is equivalent to minimizing its negative.
    obj = -sum(f_bands_psds) / sum(psd_simulated)

    obj_values.append(obj)

    return obj


def _spectral_power_loss(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
):
    """Custom loss function based on spectral power differences.

    This loss function computes the difference between target and simulated
    power spectral densities across specified frequency bands. Useful for
    optimizing steady-state oscillations generated by Poisson drives.

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
    obj_fun_kwargs : dict
        Must contain:
        - 'target_psd': array-like, target power spectral density
        - 'target_freqs': array-like, frequencies corresponding to target_psd
        - 'freq_range': tuple, (min_freq, max_freq) to focus optimization on
        - 'n_trials': int, number of trials to simulate (default: 1)
        Optional:
        - 'scale_factor': float, scaling factor for dipoles
        - 'smooth_window_len': float, smoothing window length

    Returns
    -------
    obj : float
        Mean squared error between target and simulated PSDs in the specified
        frequency range.
    """
    import numpy as np
    from scipy.signal import periodogram

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)

    n_trials = obj_fun_kwargs.get("n_trials", 1)
    dpls = simulate_dipole(new_net, tstop=tstop, n_trials=n_trials)

    # smooth & scale
    if "scale_factor" in obj_fun_kwargs:
        [dpl.scale(obj_fun_kwargs["scale_factor"]) for dpl in dpls]
    if "smooth_window_len" in obj_fun_kwargs:
        [dpl.smooth(obj_fun_kwargs["smooth_window_len"]) for dpl in dpls]

    # Average dipoles if multiple trials
    if len(dpls) > 1:
        from ..dipole import average_dipoles
        dpl = average_dipoles(dpls)
    else:
        dpl = dpls[0]

    # Compute PSD of simulated dipole
    freqs_sim, psd_sim = periodogram(dpl.data["agg"], dpl.sfreq, window="hamming")

    # Get target PSD and frequencies
    target_psd = obj_fun_kwargs["target_psd"]
    target_freqs = obj_fun_kwargs["target_freqs"]
    freq_range = obj_fun_kwargs["freq_range"]

    # Focus on specified frequency range
    freq_mask_sim = (freqs_sim >= freq_range[0]) & (freqs_sim <= freq_range[1])
    freq_mask_target = (target_freqs >= freq_range[0]) & (target_freqs <= freq_range[1])

    freqs_sim_range = freqs_sim[freq_mask_sim]
    psd_sim_range = psd_sim[freq_mask_sim]
    target_freqs_range = target_freqs[freq_mask_target]
    target_psd_range = target_psd[freq_mask_target]

    # Interpolate to common frequency grid
    common_freqs = np.linspace(freq_range[0], freq_range[1], 
                              min(len(freqs_sim_range), len(target_freqs_range)))
    
    psd_sim_interp = np.interp(common_freqs, freqs_sim_range, psd_sim_range)
    target_psd_interp = np.interp(common_freqs, target_freqs_range, target_psd_range)

    # Normalize PSDs to make comparison scale-invariant
    psd_sim_norm = psd_sim_interp / np.sum(psd_sim_interp)
    target_psd_norm = target_psd_interp / np.sum(target_psd_interp)

    # Compute mean squared error
    obj = np.mean((psd_sim_norm - target_psd_norm) ** 2)
    obj_values.append(obj)

    return obj


def _phase_coherence_loss(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
):
    """Custom loss function based on phase coherence.

    This loss function optimizes for specific phase relationships between
    different frequency components, useful for rhythmic activity optimization.

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
    obj_fun_kwargs : dict
        Must contain:
        - 'target_coherence': float, target coherence value (0-1)
        - 'freq_band': tuple, (min_freq, max_freq) for coherence analysis
        - 'n_trials': int, number of trials to simulate (default: 1)
        Optional:
        - 'scale_factor': float, scaling factor for dipoles
        - 'smooth_window_len': float, smoothing window length

    Returns
    -------
    obj : float
        Negative coherence (since optimizer minimizes). Higher coherence
        results in lower (better) objective values.
    """
    import numpy as np
    from scipy.signal import coherence

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)

    n_trials = obj_fun_kwargs.get("n_trials", 1)
    dpls = simulate_dipole(new_net, tstop=tstop, n_trials=n_trials)

    # smooth & scale
    if "scale_factor" in obj_fun_kwargs:
        [dpl.scale(obj_fun_kwargs["scale_factor"]) for dpl in dpls]
    if "smooth_window_len" in obj_fun_kwargs:
        [dpl.smooth(obj_fun_kwargs["smooth_window_len"]) for dpl in dpls]

    # For phase coherence, we need multiple trials or segments
    if len(dpls) < 2:
        # If only one trial, split into segments
        dpl = dpls[0]
        segment_length = len(dpl.data["agg"]) // 2
        if segment_length < 100:  # Need minimum length for coherence
            obj = 1.0  # Return poor score if insufficient data
            obj_values.append(obj)
            return obj
            
        signal1 = dpl.data["agg"][:segment_length]
        signal2 = dpl.data["agg"][segment_length:2*segment_length]
    else:
        # Use different trials
        signal1 = dpls[0].data["agg"]
        signal2 = dpls[1].data["agg"]
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

    # Compute coherence
    freq_band = obj_fun_kwargs["freq_band"]
    sfreq = dpls[0].sfreq
    
    freqs, coh = coherence(signal1, signal2, fs=sfreq, nperseg=min(256, len(signal1)//4))
    
    # Focus on target frequency band
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if not np.any(freq_mask):
        obj = 1.0  # Return poor score if no frequencies in range
        obj_values.append(obj)
        return obj
        
    mean_coherence = np.mean(coh[freq_mask])
    target_coherence = obj_fun_kwargs["target_coherence"]
    
    # Minimize difference from target coherence
    obj = abs(mean_coherence - target_coherence)
    obj_values.append(obj)

    return obj


def _custom_rmse_with_weights(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
):
    """Custom RMSE loss function with user-defined time weights.

    This allows users to weight different time periods differently in the
    RMSE calculation, useful for emphasizing specific events or periods.

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
    obj_fun_kwargs : dict
        Must contain:
        - 'target': Dipole object with target data
        - 'time_weights': array-like, weights for each time point
        - 'n_trials': int, number of trials to simulate (default: 1)
        Optional:
        - 'scale_factor': float, scaling factor for dipoles
        - 'smooth_window_len': float, smoothing window length

    Returns
    -------
    obj : float
        Weighted RMSE between target and simulated dipoles.
    """
    import numpy as np

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)

    n_trials = obj_fun_kwargs.get("n_trials", 1)
    dpls = simulate_dipole(new_net, tstop=tstop, n_trials=n_trials)

    # smooth & scale
    if "scale_factor" in obj_fun_kwargs:
        [dpl.scale(obj_fun_kwargs["scale_factor"]) for dpl in dpls]
    if "smooth_window_len" in obj_fun_kwargs:
        [dpl.smooth(obj_fun_kwargs["smooth_window_len"]) for dpl in dpls]

    # Average dipoles if multiple trials
    if len(dpls) > 1:
        from ..dipole import average_dipoles
        dpl = average_dipoles(dpls)
    else:
        dpl = dpls[0]

    target = obj_fun_kwargs["target"]
    time_weights = obj_fun_kwargs["time_weights"]

    # Ensure same time base
    min_len = min(len(dpl.data["agg"]), len(target.data["agg"]), len(time_weights))
    sim_data = dpl.data["agg"][:min_len]
    target_data = target.data["agg"][:min_len]
    weights = time_weights[:min_len]

    # Compute weighted RMSE
    weighted_diff = weights * (sim_data - target_data) ** 2
    obj = np.sqrt(np.mean(weighted_diff))
    obj_values.append(obj)

    return obj