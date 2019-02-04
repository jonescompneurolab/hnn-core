import os.path as op

import numpy as np
from numpy import loadtxt
from numpy.testing import assert_array_equal


def test_mne_neuron():
    """Test to check if MNE neuron does not break."""
    # small snippet of data on data branch for now. To be deleted
    # later. Data branch should have only commit so it does not
    # pollute the history.
    from mne.utils import _fetch_file
    data_url = ('https://raw.githubusercontent.com/jasmainak/'
                'mne-neuron/test_data/dpl.txt')
    _fetch_file(data_url, 'dpl.txt')
    dpl_master = loadtxt('dpl.txt')

    fname = op.join(op.dirname(__file__), '..', '..', 'data',
                    'default', 'dpl.txt')
    dpl_pr = loadtxt(fname)
    assert_array_equal(dpl_pr, dpl_master)
