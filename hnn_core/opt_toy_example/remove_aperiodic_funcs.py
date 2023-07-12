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
