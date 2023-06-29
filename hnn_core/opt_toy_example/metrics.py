from hnn_core import simulate_dipole


def _rmse_evoked(net, param_names, target_statistic, predicted_params,
                 compute_psd):
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
       compute_psd :
           ...

       Returns
       -------
       rmse : normalized RMSE between recorded and simulated dipole
    """

    # get network with predicted params
    new_net = _set_params(net, param_names, predicted_params)
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop=100, n_trials=1)[0]

    # smooth & scale (scale must be passed in by user, window length?)
    dpl.smooth(30)
    # dpl.scale(300)

    # calculate error
    rmse = np.sqrt(((dpl.data['agg'] - target_statistic)**2).sum()
                   / len(dpl.times)) / (max(target_statistic)
                                        - min(target_statistic))
    return rmse


def _rmse_rhythmic(net, param_names, psd_target_statistic, predicted_params,
                   compute_psd):
    """The objective function for evoked responses.

       Parameters
       -----------
       net : Network
       param_names : dictionary
           Parameters to change.
       psd_target_statistic : ndarray
           PSD of recorded dipole minus the aperiodic component.
       predicted_params : list
           Parameters selected by the optimizer.
       freq_range : list
           Frequency range wihtin which to optimize the response (fmin, fmax).
       compute_psd :
           ...

       Returns
       -------
       rmse : normalized RMSE between recorded and simulated dipole
    """

    # expose these
    fmin = 0.0
    fmax = 200.0

    # simulate dpl with predicted params
    new_net = _set_params(net, param_names, predicted_params)
    dpl = simulate_dipole(new_net, tstop=100, n_trials=1)[0]

    # get psd of simulated dpl
    freqs_simulated, psd_simulated = compute_psd(dpl.data['agg'], dpl.sfreq)

    # calculate error
    rmse = np.sqrt(((psd_simulated - psd_target_statistic)**2).sum()
                   / len(dpl.times)) / (max(psd_target_statistic)
                                        - min(psd_target_statistic))

    return rmse


def _rmse_poisson():
    return


def _compute_welch_psd(target_statistic, sfreq):
    """ Computes the PSD usig scipy.signal.welch.

    Parameters
    ----------
    target_statistic : ndarray
        The target dipole. psd_array_multitaper takes input signal in the
        time-domain.
    sfreq :
        ...

    Returns
    -------
    freqs : ndarray
        The frequency points in Hz.
    psd : ndarray
        Power spectral density.
    """

    from scipy.signal import welch

    freqs, psd = welch(target_statistic, fs=sfreq)

    return freqs, psd


def _compute_multitaper_psd(target_statistic, sfreq):
    """ Computes the PSD usig mne.time_frequency.psd_array_multitaper.

    Parameters
    ----------
    target_statistic : ndarray
        The target dipole. psd_array_multitaper takes input signal in the
        time-domain.
    sfreq:
        ...

    Returns
    -------
    freqs : array
        The frequency points in Hz.
    psd : ndarray
        Power spectral density.
    """

    from mne.time_frequency import psd_array_multitaper

    psd, freqs = psd_array_multitaper(target_statistic, sfreq=sfreq)

    return freqs, psd


def _remove_aperiodic_foof(target_statistic, compute_psd, sfreq):
    """Gets periodic and aperiodic parameters. Calculates the aperiodic
    PSD from the Lorentzian function. Subtracts the aperiodic PSD from the
    neural power spectra (PSD of the target statistic).

    Relies on the Fitting Oscillations and one-over-f (FOOOF) method
    (Donoghue et al 2020) to model the PSD of EEG/MEG signals as a combination
    of a periodic component (1/f) and at least one periodic component
    (oscillations).

    Parameters
    ----------
    target_statistic : ndarray
        The target dipole.
    compute_psd :
        ...
    sfreq :
        ...

    Returns
    -------
    clean_psd : ndarray
        Neural power spectra minus PSD of the aperiodic approximation.

    """

    from fooof import FOOOF
    import numpy as np

    # get PSD of target statistic
    freqs, psd = compute_psd(target_statistic, sfreq)

    # fit FOOOF model
    fm = FOOOF(aperiodic_mode='knee')
    fm.fit(freqs, psd)

    # get aperiodic parameters
    offset, knee, exponent = fm.aperiodic_params_[0],\
        fm.aperiodic_params_[1],\
        fm.aperiodic_params_[2]

    # compute aperiodic component
    aperiodic_component = offset - np.log10(knee + (psd**exponent))

    clean_psd = psd - aperiodic_component

    return clean_psd


def _remove_aperiodic_irasa(target_statistic, compute_psd, sfreq):
    """Gets periodic and aperiodic PSD components. Subtracts the aperiodic
    PSD from the neural power spectra (PSD of the target statistic).

    Relies on the Irregular-Resampling Auto-Spectral Analysis (IRASA) method
    (Wen & Liu 2016) to separate the aperiodic (1/f) and periodic components
    (oscillations) in the PSD of EEG signals.


    Parameters
    ----------
    target_statistic : ndarray
        The target dipole.
    compute_psd :
        ...
    sfreq :
        ...

    Returns
    -------
    clean_psd : ndarray
        Neural power spectra minus PSD of the aperiodic approximation.

    """

    import yasa

    # get PSD of target statistic
    freqs, psd = compute_psd(target_statistic, sfreq)

    # get aperiodic component
    f, aperiodic_component, periodic_component = yasa.irasa(data=target_statistic,
                                                            sf=sfreq,
                                                            return_fit=False)

    clean_psd = psd - aperiodic_component

    return clean_psd
