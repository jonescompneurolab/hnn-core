from pylab import convolve
from numpy import hamming
import numpy as np


def boxfilt(x, winsz):
    win = [1.0 / winsz for i in range(int(winsz))]
    return convolve(x, win, 'same')


def hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def emptyfilt(x, winsz):
    return np.array(x)
