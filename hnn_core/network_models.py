"""Network model functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

import os.path as op
import hnn_core
from hnn_core import read_params
from .network import Network
from .params import _short_name


def default_network(params=None, add_drives_from_params=False):
    """Instantiate the default network.

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

    Notes
    -----
    The default network recreates the model from Jones et al. 2009.
    """
    net = jones_2009_model(params, add_drives_from_params)
    return net


def jones_2009_model(params=None, add_drives_from_params=False):
    """Instantiate the Jones et al. 2009 model.

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

    Notes
    -----
    Network is composed of an all-to-all connectivity pattern between cells.
    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    if params is None:
        params = read_params(params_fname)

    net = Network(params, add_drives_from_params=add_drives_from_params)

    delay = net.delay

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    lamtha = 3.0
    loc = 'proximal'
    for target_cell in ['L2_pyramidal', 'L5_pyramidal']:
        for receptor in ['nmda', 'ampa']:
            key = f'gbar_{_short_name(target_cell)}_'\
                  f'{_short_name(target_cell)}_{receptor}'
            weight = net._params[key]
            net.add_connection(
                target_cell, target_cell, loc, receptor, weight,
                delay, lamtha, allow_autapses=False)

    # layer2 Basket -> layer2 Pyr
    src_cell = 'L2_basket'
    target_cell = 'L2_pyramidal'
    lamtha = 50.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L2Basket_L2Pyr_{receptor}'
        weight = net._params[key]
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer5 Basket -> layer5 Pyr
    src_cell = 'L5_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 70.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L5Basket_{_short_name(target_cell)}_{receptor}'
        weight = net._params[key]
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Pyr -> layer5 Pyr
    src_cell = 'L2_pyramidal'
    lamtha = 3.
    receptor = 'ampa'
    for loc in ['proximal', 'distal']:
        key = f'gbar_L2Pyr_{_short_name(target_cell)}'
        weight = net._params[key]
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Basket -> layer5 Pyr
    src_cell = 'L2_basket'
    lamtha = 50.
    key = f'gbar_L2Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'distal'
    receptor = 'gabaa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer2 Basket
    src_cell = 'L2_pyramidal'
    target_cell = 'L2_basket'
    lamtha = 3.
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = 'L2_basket'
    lamtha = 20.
    key = f'gbar_L2Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'gabaa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer5 Basket
    src_cell = 'L5_basket'
    target_cell = 'L5_basket'
    lamtha = 20.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha,
        allow_autapses=False)

    src_cell = 'L5_pyramidal'
    lamtha = 3.
    key = f'gbar_L5Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = 'L2_pyramidal'
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


def law_2021_model():
    """Instantiate the beta modulated ERP network model.

    Returns
    -------
    net : Instance of Network object
        Network object used to store the default network.
        All connections defining the default network will be
        appeneded to net.connectivity.
    Notes
    -----
    Model reproduces results from Law et al. 2021
    This model differs from the default network model in several
    parameters including: (fill in later)
    """

    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    params['tstop'] = 400.0
    net = jones_2009_model(params)

    # Update biophysics (increase gabab duration of inhibition)
    net.cell_types['L2_pyramidal'].p_syn['gabab']['tau1'] = 45.0
    net.cell_types['L2_pyramidal'].p_syn['gabab']['tau2'] = 200.0
    net.cell_types['L5_pyramidal'].p_syn['gabab']['tau1'] = 45.0
    net.cell_types['L5_pyramidal'].p_syn['gabab']['tau2'] = 200.0

    # Decrease L5_pyramidal -> L5_pyramidal nmda weight
    net.connectivity[2]['nc_dict']['A_weight'] = 0.0004

    # Modify L5_basket -> L5_pyramidal inhibition
    net.connectivity[6]['nc_dict']['A_weight'] = 0.02  # gabaa
    net.connectivity[7]['nc_dict']['A_weight'] = 0.005  # gabab

    # Remove L5 pyramidal somatic and basal dendrite calcium channels
    for sec in ['soma', 'basal_1', 'basal_2', 'basal_3']:
        del net.cell_types['L5_pyramidal'].p_secs[
            sec]['mechs']['ca']

    # Remove L2_basket -> L5_pyramidal gabaa connection
    del net.connectivity[10]  # Original paper simply sets gbar to 0.0

    # Add L2_basket -> L5_pyramidal gabab connection
    delay = net.delay
    src_cell = 'L2_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 50.
    weight = 0.0002
    loc = 'distal'
    receptor = 'gabab'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # Add L5_basket -> L5_pyramidal distal connection
    # ("Martinotti-like recurrent tuft connection")
    src_cell = 'L5_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 70.
    loc = 'distal'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_L5Pyr_{receptor}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net
