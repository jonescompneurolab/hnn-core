# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          George Dang <george_dang@brown.edu>

import os.path as op
import pytest

import hnn_core
from hnn_core import read_params, Network
from hnn_core.network import pick_connection

hnn_core_root = op.dirname(hnn_core.__file__)


@pytest.fixture(scope="module")
def base_network():
    """ Base Network with connections and drives """
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(params, legacy_mode=False)
    # add some basic local network connectivity
    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    for target_cell in ['L2_pyramidal', 'L5_pyramidal']:
        for receptor in ['nmda', 'ampa']:
            net.add_connection(
                target_cell, target_cell, loc='proximal', receptor=receptor,
                weight=5e-4, delay=net.delay, lamtha=3.0, allow_autapses=False)
    # layer2 Basket -> layer2 Pyr
    # layer5 Basket -> layer5 Pyr
    for receptor in ['gabaa', 'gabab']:
        net.add_connection(
            src_gids='L2_basket', target_gids='L2_pyramidal', loc='soma',
            receptor=receptor, weight=5e-4, delay=net.delay, lamtha=50.0)
        net.add_connection(
            src_gids='L5_basket', target_gids='L2_pyramidal', loc='soma',
            receptor=receptor, weight=5e-4, delay=net.delay, lamtha=70.0)
    # layer2 Basket -> layer2 Basket (autapses allowed)
    net.add_connection(
        src_gids='L2_basket', target_gids='L2_basket', loc='soma',
        receptor='gabaa', weight=5e-4, delay=net.delay, lamtha=20.0)

    # add arbitrary drives that contribute artificial cells to network
    net.add_evoked_drive(name='evdist1', mu=5.0, sigma=1.0,
                         numspikes=1, location='distal',
                         weights_ampa={'L2_basket': 0.1,
                                       'L2_pyramidal': 0.1})
    net.add_evoked_drive(name='evprox1', mu=5.0, sigma=1.0,
                         numspikes=1, location='proximal',
                         weights_ampa={'L2_basket': 0.1,
                                       'L2_pyramidal': 0.1})
    return net, params
