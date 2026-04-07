"""Utility functions."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

import numpy as np
from .externals.mne import _validate_type


def _hammfilt(x, winsz):
    """Convolve with a hamming window.

    Parameters
    ----------
    x : list | np.ndarray
        The data to filter
    winsz : int
        The size (in samples) of a `~numpy.hamming` window to convolve the data with.
        This must be an integer that is >= 1.
    """

    win = np.hamming(winsz)
    win /= np.sum(win)
    return np.convolve(x, win, "same")


def _replace_dict_identifier(input_dict, old_identifier, new_identifier):
    """Recursively replace keys and values in a dict that match an identifier.

    This takes an `old_identifier` (anything that can compared to any keys or
    values with an equivalence relation), recurses through a dictionary
    (including any nested lists or dicts), and replaces all instances of
    `old_identifier` with `new_identifier`, regardless of if they are keys or
    values.

    Parameters
    ----------
    input_dict : dict
        The dictionary to update.
    old_identifier : str
        The key or value to replace.
    new_identifier : str
        The new key or value to replace with.

    Returns
    -------
    updated_dict : dict
        The updated dictionary with keys and values replaced.
    """
    if isinstance(input_dict, dict):
        updated_dict = dict()
        for key, value in input_dict.items():
            new_key_name = new_identifier if key == old_identifier else key
            updated_dict[new_key_name] = _replace_dict_identifier(
                value, old_identifier, new_identifier
            )
        return updated_dict
    elif isinstance(input_dict, list):
        return [
            _replace_dict_identifier(element, old_identifier, new_identifier)
            for element in input_dict
        ]
    else:
        return new_identifier if input_dict == old_identifier else input_dict


# Savitzky-Golay filtering, lifted and adapted from mne-python (0.22)
def _savgol_filter(data, h_freq, sfreq):
    """Filter the data using Savitzky-Golay polynomial method.

    Parameters
    ----------
    data : array-like
        The data to filter (1D)
    h_freq : float
        Approximate high cutoff frequency in Hz. Note that this
        is not an exact cutoff, since Savitzky-Golay filtering
        is done using polynomial fits
        instead of FIR/IIR filtering. This parameter is thus used to
        determine the length of the window over which a 5th-order
        polynomial smoothing is applied.
    sfreq : float
        The sampling frequency (in Hz)

    Returns
    -------
    filt_data : array-like
        The filtered data
    """  # noqa: E501
    from scipy.signal import savgol_filter

    _validate_type(sfreq, (float, int), "sfreq")
    if sfreq <= 0.0:
        raise ValueError("Sampling frequency must be positive")
    _validate_type(h_freq, (float, int), "h_freq")
    if h_freq <= 0.0:
        raise ValueError("High cutoff frequency must be positive")

    h_freq = float(h_freq)
    if h_freq >= sfreq / 2.0:
        raise ValueError("h_freq must be less than half the sample rate")

    # savitzky-golay filtering
    window_length = (int(np.round(sfreq / h_freq)) // 2) * 2 + 1
    # loop over 'agg', 'L2', and 'L5'
    filt_data = savgol_filter(data, axis=-1, polyorder=5, window_length=window_length)
    return filt_data


def smooth_waveform(data, window_len, sfreq):
    """Smooth an arbitrary waveform using Hamming-windowed convolution

    This takes ``window_len`` ("window length" in ms) and ``sfreq`` (sampling frequency
    in Hz) to determine the ``winsz`` ("window size" in samples) of a Hamming window to
    convolve with the data. The final "window size" must be >= 1 and is defined
    by the following equation:

        winsz = 1e-3 * window_len * sfreq

    Parameters
    ----------
    data : list | np.ndarray
        The data to filter
    window_len : float
        The length (in ms) of a `~numpy.hamming` window to convolve the data with.
    sfreq : float
        The data sampling rate (in Hz).

    Returns
    -------
    data_filt : np.ndarray
        The filtered data
    """
    if (isinstance(data, np.ndarray) and data.ndim > 1) or (
        isinstance(data, list) and isinstance(data[0], list)
    ):
        raise RuntimeError("Smoothing currently only supported for 1D-arrays")

    if not isinstance(window_len, (float, int)) or window_len < 0:
        raise ValueError("Window length must be a non-negative number")
    elif 0 < window_len < 1:
        raise ValueError("Window length less than 1 ms is not supported")

    _validate_type(sfreq, (float, int), "sfreq")
    if sfreq <= 0.0:
        raise ValueError("Sampling frequency must be positive")
    # convolutional filter length is given in samples
    winsz = int(np.round(1e-3 * window_len * sfreq))
    if winsz < 1:
        # @ntolley TODO AES: I now believe this check is correct. Since winsz must be an
        # integer and cannot be negative, it can either be 0 or positive. If 0, then in
        # `_hammfilt`, the window will be an empty array and np.convolve fails since
        # then its second argument is empty, which it cannot be. Therefore, winsz must
        # be >= 1 always. This comment should be removed after review.
        raise ValueError(
            f"Window size is too small: {winsz} samples. Window size is given by "
            "(1e-3 * window_len * sfreq) and must be >= 1."
        )
    if winsz > len(data):
        raise ValueError(
            f"Window size is too long: {winsz} samples; data length is {len(data)} samples"
        )

    return _hammfilt(data, winsz)
