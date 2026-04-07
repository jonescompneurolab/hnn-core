import pytest
import numpy as np
from hnn_core.utils import smooth_waveform, _savgol_filter


def test_hamming_smoothing():
    """Test hamming window convolution smoothing"""
    # This uses our true default simulation window length of 30 ms and sampling
    # rate of 40 kHz
    default_window_len = 30
    default_sfreq = 40000

    # Our minimum window size is (1e-3 * window_len * sfreq) in samples. The size of the
    # data array must be > window size (in samples).
    smooth_waveform(
        np.arange(1e-3 * default_window_len * default_sfreq),
        default_window_len,
        default_sfreq,
    )

    # Tests of data argument
    # ---------------------------------------------------------------
    # Even one sample short should raise an error
    with pytest.raises(ValueError, match="Window size is too long"):
        smooth_waveform(
            np.arange((1e-3 * default_window_len * default_sfreq) - 1),
            default_window_len,
            default_sfreq,
        )

    # simpler values for other arguments
    test_window_len, test_sfreq = 1, 1
    # data must be 1D
    for data in [[[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])]:
        with pytest.raises(RuntimeError, match="Smoothing currently only sup"):
            smooth_waveform(data, test_window_len, test_sfreq)

    # Tests of window_len
    # ---------------------------------------------------------------
    data, test_sfreq = np.random.random((100,)), 1
    # window_len is positive number, longer than data, >1ms, and not a float
    with pytest.raises(ValueError, match="Window length must be a non-negative number"):
        smooth_waveform(data, None, test_sfreq)
    with pytest.raises(ValueError, match="Window length must be a non-negative number"):
        smooth_waveform(data, -1, test_sfreq)
    with pytest.raises(ValueError, match="Window length less than 1 ms is"):
        smooth_waveform(data, 1e-1, test_sfreq)
    with pytest.raises(ValueError, match="Window size is too long"):
        smooth_waveform(data, 1e6, test_sfreq)
    with pytest.raises(ValueError, match="Window size is too small"):
        smooth_waveform(data, 500, test_sfreq)
    # Goldlocks: Window size is just right:
    #     np.round(1e-3 * 501 * 1) = rounded to 1 sample
    smooth_waveform(data, 501, test_sfreq)

    # Tests of sfreq
    # ---------------------------------------------------------------
    # sfreq is positive number
    data, test_window_len = np.random.random((100,)), 1
    with pytest.raises(TypeError):
        smooth_waveform(data, test_window_len, None)
    with pytest.raises(TypeError):
        smooth_waveform(data, test_window_len, [1])
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        smooth_waveform(data, test_window_len, -1)
    with pytest.raises(ValueError, match="Window size is too small"):
        smooth_waveform(data, test_window_len, 500)
    # Goldlocks: Window size is just right:
    #     np.round(1e-3 * 1 * 501) = rounded to 1 sample
    smooth_waveform(data, test_window_len, 501)


def test_savgol_filter():
    """Test Savitzky-Golay smoothing"""
    data, sfreq = np.random.random((100,)), 1

    # h_freq is positive number and less than half the sampling rate
    for h_freq in [None, [1], -1, sfreq / 2]:
        with pytest.raises((TypeError, AssertionError, ValueError)):
            _savgol_filter(data, h_freq, sfreq)

    h_freq = 0.6

    # sfreq is positive number and at least twice the cutoff frequency
    for sfreq in [None, [1], -1, 1]:
        with pytest.raises((TypeError, AssertionError, ValueError)):
            _savgol_filter(data, h_freq, sfreq)
