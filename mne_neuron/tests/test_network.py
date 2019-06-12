# Authors: Mainak Jas <mainakjas@gmail.com>

import os.path as op

import mne_neuron
from mne_neuron import Params, Network


def test_network():
    """Test network object."""
    mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    params = Params(params_fname)
    net = Network(params)
    print(net)
    print(net.cells[:2])
