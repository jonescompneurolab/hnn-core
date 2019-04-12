# Authors: Mainak Jas <mainakjas@gmail.com>

import os.path as op

import mne_neuron
from mne_neuron import Params, Network


def test_params():
    """Test params object."""
    mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    params = Params(params_fname)
    print(params)

    Network(params)
    print(params['L2Pyr*'])
