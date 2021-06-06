"""Network model functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

import os.path as op
import hnn_core
from hnn_core import read_params
from .network import Network


def default_network(params=None, add_drives_from_params=False):
    """Instantiate the default all-to-all connected network.

    Parameters
    ----------
    params : dict | None
        The parameters to use for constructing the network.
        If None, parameters loaded from default.json
        Default: None
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False

    Returns
    -------
    net : Instance of Network object
        Network object used to store the default network.
        All connections defining the default network will be
        appeneded to net.connectivity.
    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'beta_erp.json')
    if params is None:
        params = read_params(params_fname)

    net = Network(params, add_drives_from_params=add_drives_from_params)

    nc_dict = {
        'A_delay': net.delay,
        'threshold': net.threshold,
    }

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    nc_dict['lamtha'] = 3.
    loc = 'proximal'
    for target_cell in ['L2Pyr', 'L5Pyr']:
        for receptor in ['nmda', 'ampa']:
            key = f'gbar_{target_cell}_{target_cell}_{receptor}'
            nc_dict['A_weight'] = net._params[key]
            net._all_to_all_connect(
                target_cell, target_cell, loc, receptor,
                nc_dict, allow_autapses=False)

    # layer2 Basket -> layer2 Pyr
    src_cell = 'L2Basket'
    target_cell = 'L2Pyr'
    nc_dict['lamtha'] = 50.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L2Basket_L2Pyr_{receptor}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer5 Basket -> layer5 Pyr
    src_cell = 'L5Basket'
    target_cell = 'L5Pyr'
    nc_dict['lamtha'] = 70.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L5Basket_{target_cell}_{receptor}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer2 Pyr -> layer5 Pyr
    src_cell = 'L2Pyr'
    nc_dict['lamtha'] = 3.
    receptor = 'ampa'
    for loc in ['proximal', 'distal']:
        key = f'gbar_L2Pyr_{target_cell}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer2 Basket -> layer5 Pyr
    src_cell = 'L2Basket'
    nc_dict['lamtha'] = 50.
    key = f'gbar_L2Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'distal'
    receptor = 'gabaa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    # xx -> layer2 Basket
    src_cell = 'L2Pyr'
    target_cell = 'L2Basket'
    nc_dict['lamtha'] = 3.
    key = f'gbar_L2Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    src_cell = 'L2Basket'
    nc_dict['lamtha'] = 20.
    key = f'gbar_L2Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'gabaa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    # xx -> layer5 Basket
    src_cell = 'L5Basket'
    target_cell = 'L5Basket'
    nc_dict['lamtha'] = 20.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict,
        allow_autapses=False)

    src_cell = 'L5Pyr'
    nc_dict['lamtha'] = 3.
    key = f'gbar_L5Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    src_cell = 'L2Pyr'
    key = f'gbar_L2Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    return net


def beta_erp_network(params=None, add_drives_from_params=False):
    """Instantiate the network model used in Law et al. 2021.

    Parameters
    ----------
    params : dict | None
        The parameters to use for constructing the network.
        If None, parameters loaded from beta_erp.json
        Default: None
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False

    Returns
    -------
    net : Instance of Network object
        Network object used to store the default network.
        All connections defining the default network will be
        appeneded to net.connectivity.
    """

    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    if params is None:
        params = read_params(params_fname)

    net = Network(params, add_drives_from_params=add_drives_from_params)

    nc_dict = {
        'A_delay': net.delay,
        'threshold': net.threshold,
    }

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    nc_dict['lamtha'] = 3.
    loc = 'proximal'
    for target_cell in ['L2Pyr', 'L5Pyr']:
        for receptor in ['nmda', 'ampa']:
            key = f'gbar_{target_cell}_{target_cell}_{receptor}'
            nc_dict['A_weight'] = net._params[key]
            net._all_to_all_connect(
                target_cell, target_cell, loc, receptor,
                nc_dict, allow_autapses=False)

    # layer2 Basket -> layer2 Pyr
    src_cell = 'L2Basket'
    target_cell = 'L2Pyr'
    nc_dict['lamtha'] = 50.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L2Basket_L2Pyr_{receptor}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer5 Basket -> layer5 Pyr
    src_cell = 'L5Basket'
    target_cell = 'L5Pyr'
    nc_dict['lamtha'] = 70.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L5Basket_{target_cell}_{receptor}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer2 Pyr -> layer5 Pyr
    src_cell = 'L2Pyr'
    nc_dict['lamtha'] = 3.
    receptor = 'ampa'
    for loc in ['proximal', 'distal']:
        key = f'gbar_L2Pyr_{target_cell}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer2 Basket -> layer5 Pyr
    src_cell = 'L2Basket'
    nc_dict['lamtha'] = 50.
    key = f'gbar_L2Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'distal'
    receptor = 'gabaa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    # xx -> layer2 Basket
    src_cell = 'L2Pyr'
    target_cell = 'L2Basket'
    nc_dict['lamtha'] = 3.
    key = f'gbar_L2Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    src_cell = 'L2Basket'
    nc_dict['lamtha'] = 20.
    key = f'gbar_L2Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'gabaa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    # xx -> layer5 Basket
    src_cell = 'L5Basket'
    target_cell = 'L5Basket'
    nc_dict['lamtha'] = 20.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict,
        allow_autapses=False)

    src_cell = 'L5Pyr'
    nc_dict['lamtha'] = 3.
    key = f'gbar_L5Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    src_cell = 'L2Pyr'
    key = f'gbar_L2Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    return net
