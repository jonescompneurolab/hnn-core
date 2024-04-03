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


@pytest.mark.parametrize("arg_name",
                         ["src_gids", "target_gids", "loc", "receptor"]
                         )
def test_pc_1arg_none(base_network, arg_name):
    """ Tests passing None as an argument value. """
    net, _ = base_network
    kwargs = {'net': net, f'{arg_name}': None}
    indices = pick_connection(**kwargs)
    assert not indices


@pytest.mark.parametrize("arg_name", ["src_gids", "target_gids"])
def test_pc_1arg_range(base_network, arg_name):
    """ Tests passing range as an argument value. """
    net, _ = base_network
    test_range = range(2)
    kwargs = {'net': net, f'{arg_name}': test_range}
    indices = pick_connection(**kwargs)

    for conn_idx in indices:
        assert set(test_range).issubset(net.connectivity[conn_idx][arg_name])


@pytest.mark.parametrize("arg_name,value",
                         [("src_gids", 'L2_pyramidal'),
                          ("target_gids", 'L2_pyramidal'),
                          ("loc", 'soma'),
                          ("receptor", 'gabaa'),
                          ])
def test_pc_1arg_str(base_network, arg_name, value):
    """ Tests passing string as an argument value. """
    net, _ = base_network
    kwargs = {'net': net, f'{arg_name}': value}
    indices = pick_connection(**kwargs)

    for conn_idx in indices:
        if arg_name in ('src_gids', 'target_gids'):
            # arg specifies a subset of item gids (within gid_ranges)
            assert (net.connectivity[conn_idx][arg_name]
                    .issubset(net.gid_ranges[value])
                    )
        else:
            # arg and item specify equivalent string descriptors
            assert net.connectivity[conn_idx][arg_name] == value


@pytest.mark.parametrize("arg_name,value",
                         [("src_gids", 0),
                          ("target_gids", 35),
                          ])
def test_pc_1arg_int(base_network, arg_name, value):
    """
    Pick_connection is not missing qualifying connections with single gid.
    """
    net, _ = base_network
    kwargs = {'net': net, f'{arg_name}': value}
    indices = pick_connection(**kwargs)

    for conn_idx in range(len(net.connectivity)):
        if conn_idx in indices:
            assert value in net.connectivity[conn_idx][arg_name]
        else:
            assert value not in net.connectivity[conn_idx][arg_name]


@pytest.mark.parametrize("arg_name,value",
                         [("src_gids", ['L2_basket', 'L5_basket']),
                          ("target_gids", ['L2_pyramidal', 'L5_pyramidal']),
                          ])
def test_pc_1arg_list_str(base_network, arg_name, value):
    """ Tests passing a list of valid strings """
    net, _ = base_network
    kwargs = {'net': net, f'{arg_name}': value}
    indices = pick_connection(**kwargs)

    true_gid_set = set(list(net.gid_ranges[value[0]]) +
                       list(net.gid_ranges[value[1]])
                       )
    pick_gid_list = []
    for idx in indices:
        pick_gid_list.extend(net.connectivity[idx][arg_name])
    assert true_gid_set == set(pick_gid_list)


@pytest.mark.parametrize("src_gids,target_gids,loc,receptor",
                         [("evdist1", None, "proximal", None),
                          ("evprox1", None, "distal", None),
                          (None, None, "distal", "gabab"),
                          ("L2_pyramidal", None, None, "gabab"),
                          ("L2_basket", "L2_basket", "proximal", "nmda"),
                          ("L2_pyramidal", "L2_basket", "distal", "gabab"),
                          ])
def test_pc_no_match(base_network,
                     src_gids, target_gids, loc, receptor):
    """ Tests no matches returned for non-configured connections. """
    net, _ = base_network
    indices = pick_connection(net,
                              src_gids=src_gids,
                              target_gids=target_gids,
                              loc=loc,
                              receptor=receptor)
    assert len(indices) == 0


@pytest.mark.parametrize("src_gids,target_gids,loc,receptor",
                         [(0.0, None, None, None),
                          ([0.0], None, None, None),
                          (None, 35.0, None, None),
                          (None, [35.0], None, None),
                          (None, [35, [36.0]], None, None),
                          (None, None, 1.0, None),
                          (None, None, None, 1.0),
                          ])
def test_pc_type_error(base_network,
                       src_gids, target_gids, loc, receptor):
    """ Tests TypeError when passing floats. """
    net, _ = base_network
    match = ('must be an instance of')
    with pytest.raises(TypeError, match=match):
        pick_connection(net,
                        src_gids=src_gids,
                        target_gids=target_gids,
                        loc=loc,
                        receptor=receptor)
