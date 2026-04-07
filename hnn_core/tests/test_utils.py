import pytest
import numpy as np
from hnn_core.utils import smooth_waveform, _savgol_filter


def test_hamming_smoothing():
    """Test hamming window convolution smoothing"""
    # This uses our true default simulation window length of 30 samples and sampling
    # rate of 40 kHz
    default_window_len = 30
    default_sfreq = 40000

    # Our minimum window is (1e-3 * window_len * sfreq) in samples. Length of data must
    # be > window size (in samples).
    #
    # @ntolley TODO it turns out that our original tests did NOT actually include a test
    # of successful `smooth_waveform` execution, and instead only tested failure modes.
    # This adds a test of successful execution. This comment should be removed after
    # review.
    smooth_waveform(
        np.arange(1e-3 * default_window_len * default_sfreq),
        default_window_len,
        default_sfreq,
    )

    # Tests of data argument
    # ---------------------------------------------------------------
    # Even one sample short should raise an error
    with pytest.raises(ValueError):
        smooth_waveform(
            np.arange((1e-3 * default_window_len * default_sfreq) - 1),
            default_window_len,
            default_sfreq,
        )

    # simpler values for other arguments
    test_window_len, test_sfreq = 1, 1
    # data must be 1D
    for data in [[[1, 2], [3, 4]], np.array([[1, 2], [3, 4]])]:
        with pytest.raises(RuntimeError):
            smooth_waveform(data, test_window_len, test_sfreq)

    # Tests of window_len
    # ---------------------------------------------------------------
    data, test_sfreq = np.random.random((100,)), 1
    # window_len is positive number, longer than data, >1ms, and not a float
    for test_window_len in [None, -1, 1e6, 1e-1, 2.5]:
        with pytest.raises(ValueError):
            smooth_waveform(data, test_window_len, test_sfreq)

    # Tests of sfreq
    # ---------------------------------------------------------------
    # sfreq is positive number
    data, test_window_len = np.random.random((100,)), 1
    for test_sfreq in [None, [1], -1]:
        with pytest.raises((TypeError, AssertionError, ValueError)):
            smooth_waveform(data, test_window_len, test_sfreq)


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
