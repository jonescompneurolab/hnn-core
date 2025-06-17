"""Network model functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

import os.path as op
import hnn_core
from hnn_core import read_params
from .network import Network
from .params import _short_name
from .cells_default import pyramidal_ca
from .externals.mne import _validate_type
from .cells_default import pyramidal, basket
import itertools as it
import numpy as np


def _create_cell_coords(n_pyr_x, n_pyr_y, zdiff, inplane_distance):
    """Creates coordinate grid for different cortical layers.

    Parameters
    ----------
    n_pyr_x : int
        The number of cells in x direction.
    n_pyr_y : int
        The number of cells in y direction.
    zdiff : float
        Expressed as a positive DEPTH of L2 relative to L5 cell
        somas, where L5 is defined to lie at z==0.
    inplane_distance : float
        The grid spacing of cells (in um).

    Returns
    -------
    layer_dict : dict of list of tuple (x, y, z)
        Dictionary containing coordinate positions for each layer.
        Keys are 'L5_bottom', 'L2_bottom', 'L5_mid', 'L2_mid', 'origin'
    """
    def _calc_grid_positions(xxrange, yyrange, z_coord):
        """Calculate positions on a regular grid at given z-coordinate."""
        list_coords = [pos for pos in it.product(xxrange, yyrange, [z_coord])]
        return list_coords

    def _calc_basket_positions(n_x, n_y, z_coord, inplane_distance):
        """Calculate basket cell positions in checkerboard pattern."""
        xzero = np.arange(0, n_x, 3) * inplane_distance
        xone = np.arange(1, n_x, 3) * inplane_distance
        # split even and odd y vals
        yeven = np.arange(0, n_y, 2) * inplane_distance
        yodd = np.arange(1, n_y, 2) * inplane_distance
        # create general list of x,y coords and sort it
        coords = [pos for pos in it.product(xzero, yeven)] + \
                 [pos for pos in it.product(xone, yodd)]
        coords_sorted = sorted(coords, key=lambda pos: pos[1])

        # append the z value for position
        list_coords = [(pos_xy[0], pos_xy[1], z_coord)
                       for pos_xy in coords_sorted]
        return list_coords

    def _calc_origin(xxrange, yyrange, zdiff):
        """Calculate the origin position."""
        origin_x = xxrange[int((len(xxrange) - 1) // 2)]
        origin_y = yyrange[int((len(yyrange) - 1) // 2)]
        origin_z = np.floor(zdiff / 2)
        origin = (origin_x, origin_y, origin_z)
        return origin

    # Calculate coordinate ranges
    xxrange = np.arange(n_pyr_x) * inplane_distance
    yyrange = np.arange(n_pyr_y) * inplane_distance

    # Create layer dictionary with anatomical layer positions
    layer_dict = {
        'L5_bottom': _calc_grid_positions(xxrange, yyrange, z_coord=0),
        'L2_bottom': _calc_grid_positions(xxrange, yyrange, z_coord=zdiff),
        'L5_mid': _calc_basket_positions(n_pyr_x, n_pyr_y, 
                                         z_coord=0.2 * zdiff, 
                                         inplane_distance=inplane_distance),
        'L2_mid': _calc_basket_positions(n_pyr_x, n_pyr_y, 
                                         z_coord=0.8 * zdiff, 
                                         inplane_distance=inplane_distance),
        'origin': _calc_origin(xxrange, yyrange, zdiff),
    }

    return layer_dict

def _compare_lists(s, t):
    """
    Compares lists for equality

    From https://stackoverflow.com/a/7829388
    """
    t = list(t)  # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t


def jones_2009_model(params=None, add_drives_from_params=False,
                     legacy_mode=False, mesh_shape=(10, 10), 
                     custom_positions=None):
    """Instantiate the network model described in
    Jones et al. J. of Neurophys. 2009 [1]_

    Parameters
    ----------
    params : str | dict | None
        The path to the parameter file for constructing the network.
        If None, parameters loaded from default.json
        Default: None
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.

    Returns
    -------
    net : Instance of Network object
        Network object used to store

    Notes
    -----
    The network is composed of a square grid of pyramidal cells, arranged in
    two layers (L5 and L2). The default in-plane separation of the grid points
    is 1.0 um, and the layer separation 1307.4 um. These can be adjusted after
    the net is created using the set_cell_positions-method. An all-to-all
    connectivity pattern is applied between cells. Inhibitory basket cells are
    present at a 1:3-ratio.

    References
    ----------
    .. [1] Jones, Stephanie R., et al. "Quantitative Analysis and
           Biophysically Realistic Neural Modeling of the MEG Mu Rhythm:
           Rhythmogenesis and Modulation of Sensory-Evoked Responses."
           Journal of Neurophysiology 102, 3554–3572 (2009).

    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    if params is None:
        params = op.join(hnn_core_root, 'param', 'default.json')
    if isinstance(params, str):
        params = read_params(params)

    # Define cell types for Jones 2009 model
    cell_types = {
        'L2_basket': basket(cell_name=_short_name('L2_basket')),
        'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
        'L5_basket': basket(cell_name=_short_name('L5_basket')),
        'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal'))
    }
    
    # Create layer positions
    layer_dict = _create_cell_coords(
        n_pyr_x=mesh_shape[0],
        n_pyr_y=mesh_shape[1],
        zdiff=1307.4,  # Default layer separation
        inplane_distance=1.0  # Default in-plane distance
    )
    
    # Map cell types to layer positions
    pos_dict = {
        'L5_pyramidal': layer_dict['L5_bottom'],
        'L2_pyramidal': layer_dict['L2_bottom'],
        'L5_basket': layer_dict['L5_mid'],
        'L2_basket': layer_dict['L2_mid'],
        'origin': layer_dict['origin']
    }
    
    # Create network with cell types and positions
    net = Network(params, 
                  add_drives_from_params=add_drives_from_params,
                  legacy_mode=legacy_mode, 
                  pos_dict=pos_dict,
                  cell_types=cell_types)
    
    delay = net.delay

    # connectivity refactored
    connectivity = {
        'L2_pyramidal': {
            'L2_pyramidal': [
                {
                    'receptor': 'nmda',
                    'weight': net._params['gbar_L2Pyr_L2Pyr_nmda'],  # 0.0005
                    'loc': 'proximal',
                    'lamtha': 3.0,
                    'allow_autapses': False
                },
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L2Pyr_L2Pyr_ampa'],  # 0.0005
                    'loc': 'proximal',
                    'lamtha': 3.0,
                    'allow_autapses': False
                }
            ],
            'L5_pyramidal': [
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L2Pyr_L5Pyr'],  # 0.00025
                    'loc': 'proximal',
                    'lamtha': 3.0,
                    'allow_autapses': True
                },
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L2Pyr_L5Pyr'],  # 0.00025
                    'loc': 'distal',
                    'lamtha': 3.0,
                    'allow_autapses': True
                }
            ],
            'L2_basket': [
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L2Pyr_L2Basket'],  # 0.0005
                    'loc': 'soma',
                    'lamtha': 3.0,
                    'allow_autapses': True
                }
            ],
            'L5_basket': [
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L2Pyr_L5Basket'],  # 0.00025
                    'loc': 'soma',
                    'lamtha': 3.0,
                    'allow_autapses': True
                }
            ]
        },
        'L2_basket': {
            'L2_pyramidal': [
                {
                    'receptor': 'gabaa',
                    'weight': net._params['gbar_L2Basket_L2Pyr_gabaa'],  # 0.05
                    'loc': 'soma',
                    'lamtha': 50.0,
                    'allow_autapses': True
                },
                {
                    'receptor': 'gabab',
                    'weight': net._params['gbar_L2Basket_L2Pyr_gabab'],  # 0.05
                    'loc': 'soma',
                    'lamtha': 50.0,
                    'allow_autapses': True
                }
            ],
            'L5_pyramidal': [
                {
                    'receptor': 'gabaa',
                    'weight': net._params['gbar_L2Basket_L5Pyr'],  # 0.001
                    'loc': 'distal',
                    'lamtha': 50.0,
                    'allow_autapses': True
                }
            ],
            'L2_basket': [
                {
                    'receptor': 'gabaa',
                    'weight': net._params['gbar_L2Basket_L2Basket'],  # 0.02
                    'loc': 'soma',
                    'lamtha': 20.0,
                    'allow_autapses': True
                }
            ]
        },
        'L5_pyramidal': {
            'L5_pyramidal': [
                {
                    'receptor': 'nmda',
                    'weight': net._params['gbar_L5Pyr_L5Pyr_nmda'],  # 0.0005
                    'loc': 'proximal',
                    'lamtha': 3.0,
                    'allow_autapses': False
                },
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L5Pyr_L5Pyr_ampa'],  # 0.0005
                    'loc': 'proximal',
                    'lamtha': 3.0,
                    'allow_autapses': False
                }
            ],
            'L5_basket': [
                {
                    'receptor': 'ampa',
                    'weight': net._params['gbar_L5Pyr_L5Basket'],  # 0.0005
                    'loc': 'soma',
                    'lamtha': 3.0,
                    'allow_autapses': True
                }
            ]
        },
        'L5_basket': {
            'L5_pyramidal': [
                {
                    'receptor': 'gabaa',
                    'weight': net._params['gbar_L5Basket_L5Pyr_gabaa'],  # 0.025
                    'loc': 'soma',
                    'lamtha': 70.0,
                    'allow_autapses': True
                },
                {
                    'receptor': 'gabab',
                    'weight': net._params['gbar_L5Basket_L5Pyr_gabab'],  # 0.025
                    'loc': 'soma',
                    'lamtha': 70.0,
                    'allow_autapses': True
                }
            ],
            'L5_basket': [
                {
                    'receptor': 'gabaa',
                    'weight': net._params['gbar_L5Basket_L5Basket'],  # 0.02
                    'loc': 'soma',
                    'lamtha': 20.0,
                    'allow_autapses': False
                }
            ]
        }
    }

    # Create connections from the connectivity dictionary
    for src_cell in connectivity:
        for target_cell in connectivity[src_cell]:
            for conn_params in connectivity[src_cell][target_cell]:
                net.add_connection(
                    src_gids=src_cell,
                    target_gids=target_cell,
                    loc=conn_params['loc'],
                    receptor=conn_params['receptor'],
                    weight=conn_params['weight'],
                    delay=delay,
                    lamtha=conn_params['lamtha'],
                    allow_autapses=conn_params.get('allow_autapses', True)
                )

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    lamtha = 3.0
    loc = 'proximal'
    for target_cell in ['L2_pyramidal', 'L5_pyramidal']:
        for receptor in ['nmda', 'ampa']:
            key = (
                f'gbar_{_short_name(target_cell)}_{_short_name(target_cell)}_{receptor}'
            )
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
    lamtha = 3.
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


def law_2021_model(params=None, add_drives_from_params=False,
                   legacy_mode=False, mesh_shape=(10, 10)):
    """Instantiate the expansion of Jones 2009 model to study beta
    modulated ERPs as described in
    Law et al. Cereb. Cortex 2021 [1]_

    Returns
    -------
    net : Instance of Network object
        Network object used to store the model used in
        Law et al. 2021.

    See Also
    --------
    jones_2009_model

    Notes
    -----
    Model reproduces results from Law et al. 2021
    This model differs from the default network model in several
    parameters including
    1) Increased GABAb time constants on L2/L5 pyramidal cells
    2) Decrease L5_pyramidal -> L5_pyramidal nmda weight
    3) Modified L5_basket -> L5_pyramidal inhibition weights
    4) Removal of L5 pyramidal somatic and basal dendrite calcium channels
    5) Replace L2_basket -> L5_pyramidal GABAa connection with GABAb
    6) Addition of L5_basket -> L5_pyramidal distal connection

    References
    ----------
    .. [1] Law, Robert G., et al. "Thalamocortical Mechanisms Regulating the
           Relationship between Transient Beta Events and Human Tactile
           Perception." Cerebral Cortex, 32, 668–688 (2022).
    """

    net = jones_2009_model(params, add_drives_from_params, legacy_mode,
                           mesh_shape=mesh_shape)

    # Update biophysics (increase gabab duration of inhibition)
    net.cell_types['L2_pyramidal'].synapses['gabab']['tau1'] = 45.0
    net.cell_types['L2_pyramidal'].synapses['gabab']['tau2'] = 200.0
    net.cell_types['L5_pyramidal'].synapses['gabab']['tau1'] = 45.0
    net.cell_types['L5_pyramidal'].synapses['gabab']['tau2'] = 200.0

    # Decrease L5_pyramidal -> L5_pyramidal nmda weight
    net.connectivity[2]['nc_dict']['A_weight'] = 0.0004

    # Modify L5_basket -> L5_pyramidal inhibition
    net.connectivity[6]['nc_dict']['A_weight'] = 0.02  # gabaa
    net.connectivity[7]['nc_dict']['A_weight'] = 0.005  # gabab

    # Remove L5 pyramidal somatic and basal dendrite calcium channels
    for sec in ['soma', 'basal_1', 'basal_2', 'basal_3']:
        del net.cell_types['L5_pyramidal'].sections[
            sec].mechs['ca']

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


# Remove params argument after updating examples
# (only relevant for Jones 2009 model)
def calcium_model(params=None, add_drives_from_params=False,
                  legacy_mode=False, mesh_shape=(10, 10)):
    """Instantiate the Jones 2009 model with improved calcium dynamics in
    L5 pyramidal neurons. For more details on changes to calcium dynamics
    see Kohl et al. Brain Topragr 2022 [1]_

    Returns
    -------
    net : Instance of Network object
        Network object used to store the Jones 2009 model with an improved
        calcium channel distribution.

    See Also
    --------
    jones_2009_model

    Notes
    -----
    This model builds on the Jones 2009 model by using a more biologically
    accurate distribution of calcium channels on L5 pyramidal cells.
    Specifically, this model introduces a distance dependent maximum
    conductance (gbar) on calcium channels such that the gbar linearly
    decreases along the dendrites in the direction of the soma.

    References
    ----------
    .. [1] Kohl, Carmen, et al. "Neural Mechanisms Underlying Human Auditory
           Evoked Responses Revealed By Human Neocortical Neurosolver."
           Brain Topography, 35, 19–35 (2022).
    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    if params is None:
        params = read_params(params_fname)

    net = jones_2009_model(params, add_drives_from_params, legacy_mode,
                           mesh_shape=mesh_shape)

    # Replace L5 pyramidal cell template with updated calcium
    cell_name = 'L5_pyramidal'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = pyramidal_ca(
        cell_name=_short_name(cell_name), pos=pos)

    return net


def add_erp_drives_to_jones_model(net, tstart=0.0):
    """Add drives necessary for an event related potential (ERP)

    Parameters
    ----------
    net : Instance of Network object
        Network object that will be updated with ERP drives.
        Drives are updated in place.
    tstart : float | int
        Start time of sensory input in ms. (Default 0.0 ms)

    Notes
    -----
    The first proximal input arrives at cortex ~20 ms after sensory
    stimulus. The exact delay depends random number generator due to
    random sampling of times from a gaussian.
    """
    _validate_type(net, Network, 'net', 'Network')
    _validate_type(tstart, (float, int), 'tstart', 'float or int')

    # Add distal drive
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': 7e-6,
                       'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        'evdist1', mu=63.53 + tstart, sigma=3.85, numspikes=1,
        weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
        location='distal', synaptic_delays=synaptic_delays_d1, event_seed=274)

    # Add proximal drives
    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                       'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    net.add_evoked_drive(
        'evprox1', mu=26.61 + tstart, sigma=2.47, numspikes=1,
        weights_ampa=weights_ampa_p1, weights_nmda=None, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=544)

    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                       'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    net.add_evoked_drive(
        'evprox2', mu=137.12 + tstart, sigma=8.33, numspikes=1,
        weights_ampa=weights_ampa_p2, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=814)
