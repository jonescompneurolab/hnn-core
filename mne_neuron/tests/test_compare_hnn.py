import os.path as op

from numpy import loadtxt
from numpy.testing import assert_array_equal

from mne.utils import _fetch_file
import mne_neuron
from mne_neuron import simulate_dipole, Params, Network


def test_mne_neuron():
    """Test to check if MNE neuron does not break."""
    # small snippet of data on data branch for now. To be deleted
    # later. Data branch should have only commit so it does not
    # pollute the history.
    data_url = ('https://raw.githubusercontent.com/jasmainak/'
                'mne-neuron/test_data/dpl.txt')
    if not op.exists('dpl.txt'):
        _fetch_file(data_url, 'dpl.txt')
    dpl_master = loadtxt('dpl.txt')

    mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    params = Params(params_fname)

    net = Network(params, n_jobs=1)
    dpl = simulate_dipole(net)

    fname = './dpl2.txt'
    dpl.write(fname)

    dpl_pr = loadtxt(fname)
    assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
    assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5
