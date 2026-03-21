import pytest
import numpy as np
from hnn_core.utils import smooth_waveform, _hammfilt, _savgol_filter

def test_hamming_smoothing():
    """Test hamming window convolution smoothing"""
    # length of data must be > window size (in samples)
    with pytest.raises(ValueError, match="winsz must be smaller than input length"):
        _hammfilt(np.arange(10), 11)

    # invalid x type
    with pytest.raises(TypeError):
        _hammfilt("not_array", 5)

    # invalid winsz type
    with pytest.raises(TypeError):
        _hammfilt(np.arange(10), "five")

    # negative winsz
    with pytest.raises(ValueError):
        _hammfilt(np.arange(10), -1)

    window_len, sfreq = 1, 1
    # data must be 1D
    for data in [[[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])]:
        pytest.raises(RuntimeError, smooth_waveform, data, window_len, sfreq)

    # window_len is positive number, longer than data, and >1ms
    data, sfreq = np.random.random((100,)), 1
    for window_len in [None, -1, 1e6, 1e-1]:
        pytest.raises(ValueError, smooth_waveform, data, window_len, sfreq)

    # sfreq is positive number
    data, window_len = np.random.random((100,)), 1
    for sfreq in [None, [1], -1]:
        pytest.raises(
            (TypeError, AssertionError), smooth_waveform, data, window_len, sfreq
        )


def test_savgol_filter():
    """Test Savitzky-Golay smoothing"""
    data, sfreq = np.random.random((100,)), 1

    # h_freq is positive number and less than half the sampling rate
    for h_freq in [None, [1], -1, sfreq / 2]:
        pytest.raises(
            (TypeError, AssertionError, ValueError), _savgol_filter, data, h_freq, sfreq
        )

    h_freq = 0.6
    # sfreq is positive number and at least twice the cutoff frequency
    for sfreq in [None, [1], -1, 1]:
        pytest.raises(
            (TypeError, AssertionError, ValueError), _savgol_filter, data, h_freq, sfreq
        )

def test_hammfilt_basic():
    x = np.array([1, 2, 3, 4, 5])
    result = _hammfilt(x, 3)
    assert len(result) == len(x)

def test_hammfilt_invalid_winsz():
    x = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _hammfilt(x, 5)

def test_hammfilt_invalid_type():
    with pytest.raises(TypeError):
        _hammfilt("abc", 3)

def test_hammfilt_zero_window():
    x = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _hammfilt(x, 0)

def test_hammfilt_float_winsz():
    x = np.array([1, 2, 3, 4])
    result = _hammfilt(x, 2.5)
    assert len(result) == len(x)
