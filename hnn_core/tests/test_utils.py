import pytest
import numpy as np
from hnn_core.utils import smooth_waveform, _savgol_filter


def test_hamming_smoothing():
    """Test hamming window convolution smoothing"""
    with pytest.raises(ValueError, match="Window length too long.*"):
        smooth_waveform(np.arange(10), 11000, 1)

    window_len, sfreq = 1, 1
    # data must be 1D
    for data in [[[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])]:
        with pytest.raises(RuntimeError):
            smooth_waveform(data, window_len, sfreq)

    # window_len is positive number, longer than data, and >1ms
    data, sfreq = np.random.random((100,)), 1
    for window_len in [-1, 1e6, 1e-1]:
        with pytest.raises(ValueError):
            smooth_waveform(data, window_len, sfreq)

    # sfreq is positive number
    data, window_len = np.random.random((100,)), 1
    for sfreq in [-1]:
        with pytest.raises(AssertionError):
            smooth_waveform(data, window_len, sfreq)


def test_savgol_filter():
    """Test Savitzky-Golay smoothing"""
    data, sfreq = np.random.random((100,)), 1

    # h_freq is positive number and less than half the sampling rate
    # negative h_freq → AssertionError
    with pytest.raises(AssertionError):
        _savgol_filter(data, -1, sfreq)

    # too large h_freq → ValueError
    with pytest.raises(ValueError):
        _savgol_filter(data, sfreq / 2, sfreq)

    h_freq = 0.6
    # sfreq is positive number and at least twice the cutoff frequency
    # invalid sfreq (<=0) → AssertionError
    with pytest.raises(AssertionError):
        _savgol_filter(data, h_freq, -1)

    # too small sfreq → ValueError
    with pytest.raises(ValueError):
        _savgol_filter(data, h_freq, 1)


def test_smooth_window_float_winsz():
    x = np.array([1, 2, 3, 4])
    result = smooth_waveform(x, 2.5, 1)
    assert len(result) == len(x)
