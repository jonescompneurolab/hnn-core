"""Network io"""
# Authors: Rajat Partani <rajatpartani@gmail.com>
#          Mainak Jas <mjas@mgh.harvard.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          George Dang <george_dang@brown.edu>

import os
import json
import numpy as np

from collections import OrderedDict

from .cell import Cell, Section
from .cell_response import CellResponse
from .externals.mne import fill_doc


def _convert_np_array_to_list(obj):
    """Returns object with np.arrays converted to lists

    Converts np.arrays to lists. Dicts or lists with nested np.arrays will
    have nested arrays converted to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_np_array_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_np_array_to_list(item) for item in obj]
    else:
        return obj


def _cell_response_to_dict(net, write_output):
    """Returns a dict of cell response data."""
    # Write cell_response as dict
    if (not net.cell_response) or (not write_output):
        return dict()
    else:
        return net.cell_response.to_dict()


def _rec_array_to_dict(value, write_output):
    """Returns a dict of rec_array data."""
    rec_array_copy = value.copy()
    if not write_output:
        rec_array_copy._reset()
    rec_array_copy_dict = rec_array_copy.to_dict()
    return rec_array_copy_dict


def _conn_to_dict(conn):
    """Converts a Connectivity object parameters to a dict format."""
    conn_data = {
        'target_type': conn['target_type'],
        'target_gids': list(conn['target_gids']),
        'num_targets': conn['num_targets'],
        'src_type': conn['src_type'],
        'src_gids': list(conn['src_gids']),
        'num_srcs': conn['num_srcs'],
        'gid_pairs': {str(key): val
                      for key, val in conn['gid_pairs'].items()},
        'loc': conn['loc'],
        'receptor': conn['receptor'],
        'nc_dict': conn['nc_dict'],
        'allow_autapses': int(conn['allow_autapses']),
        'probability': conn['probability'],
    }
    return conn_data


def _external_drive_to_dict(drive, write_output):
    """Returns dict of drive data from a Drive object."""
    drive_data = dict()
    for key in drive.keys():
        # Cannot store sets with hdf5
        if isinstance(drive[key], set):
            drive_data[key] = list(drive[key])
        else:
            drive_data[key] = drive[key]
    if not write_output:
        drive_data['events'] = list()
    return drive_data


def _str_to_node(node_string):
    """Returns tuple of node values from a comma-separated string format."""
    node_tuple = node_string.split(',')
    node_tuple[1] = int(node_tuple[1])
    node = (node_tuple[0], node_tuple[1])
    return node


def _read_cell_types(cell_types_data):
    """Returns a dict of Cell objects from json encoded data"""
    cell_types = dict()
    for cell_name in cell_types_data:
        cell_data = cell_types_data[cell_name]
        sections = dict()
        sections_data = cell_data['sections']
        for section_name in sections_data:
            section_data = sections_data[section_name]
            sections[section_name] = Section(L=section_data['L'],
                                             diam=section_data['diam'],
                                             cm=section_data['cm'],
                                             Ra=section_data['Ra'],
                                             end_pts=section_data['end_pts'])
            # Set section attributes
            sections[section_name].syns = section_data['syns']
            sections[section_name].mechs = section_data['mechs']
        # cell_tree
        cell_tree = None
        if cell_data['cell_tree'] is not None:
            cell_tree = dict()
            for parent, children in cell_data['cell_tree'].items():
                key = _str_to_node(parent)
                value = list()
                for child in children:
                    value.append(_str_to_node(child))
                cell_tree[key] = value

        cell_types[cell_name] = Cell(name=cell_data['name'],
                                     pos=tuple(cell_data['pos']),
                                     sections=sections,
                                     synapses=cell_data['synapses'],
                                     cell_tree=cell_tree,
                                     sect_loc=cell_data['sect_loc'],
                                     gid=cell_data['gid'])
        # Setting cell attributes
        cell_types[cell_name].dipole_pp = cell_data['dipole_pp']
        cell_types[cell_name].vsec = cell_data['vsec']
        cell_types[cell_name].isec = cell_data['isec']
        cell_types[cell_name].tonic_biases = cell_data['tonic_biases']

    return cell_types


def _read_cell_response(cell_response_data, read_output):
    """Returns CellResponse from json encoded data"""
    if (not cell_response_data) or (not read_output):
        return None
    cell_response = CellResponse(spike_times=cell_response_data['spike_times'],
                                 spike_gids=cell_response_data['spike_gids'],
                                 spike_types=cell_response_data['spike_types'])

    cell_response._times = cell_response_data['times']
    cell_response._vsec = list()
    for trial in cell_response_data['vsec']:
        trial = dict((int(key), val) for key, val in trial.items())
        cell_response._vsec.append(trial)
    cell_response._isec = list()
    for trial in cell_response_data['isec']:
        trial = dict((int(key), val) for key, val in trial.items())
        cell_response._isec.append(trial)
    return cell_response


def _set_from_cell_specific(drive_data):
    """Returns number of drive cells based on cell_specific bool

    The n_drive_cells keyword for add_poisson_drive and add_bursty_drive
    methods accept either an int or string (n_cells). If the bool keyword
    cell_specific = True, n_drive_cells must be 'n_cells'.
    """
    if drive_data['cell_specific']:
        return 'n_cells'
    return drive_data['n_drive_cells']


def _read_external_drive(net, drive_data, read_drives, read_output):
    """Adds drives encoded in json data to a Network"""
    if not read_drives:
        return None

    if (drive_data['type'] == 'evoked') or (drive_data['type'] == 'gaussian'):
        # Skipped n_drive_cells here
        net.add_evoked_drive(name=drive_data['name'],
                             mu=drive_data['dynamics']['mu'],
                             sigma=drive_data['dynamics']['sigma'],
                             numspikes=drive_data['dynamics']['numspikes'],
                             location=drive_data['location'],
                             cell_specific=drive_data['cell_specific'],
                             weights_ampa=drive_data['weights_ampa'],
                             weights_nmda=drive_data['weights_nmda'],
                             synaptic_delays=drive_data['synaptic_delays'],
                             probability=drive_data["probability"],
                             event_seed=drive_data['event_seed'],
                             conn_seed=drive_data['conn_seed'])
    elif drive_data['type'] == 'poisson':
        net.add_poisson_drive(name=drive_data['name'],
                              tstart=drive_data['dynamics']['tstart'],
                              tstop=drive_data['dynamics']['tstop'],
                              rate_constant=(drive_data['dynamics']
                                                       ['rate_constant']),
                              location=drive_data['location'],
                              n_drive_cells=(
                                  _set_from_cell_specific(drive_data)),
                              cell_specific=drive_data['cell_specific'],
                              weights_ampa=drive_data['weights_ampa'],
                              weights_nmda=drive_data['weights_nmda'],
                              synaptic_delays=drive_data['synaptic_delays'],
                              probability=drive_data["probability"],
                              event_seed=drive_data['event_seed'],
                              conn_seed=drive_data['conn_seed'])
    elif drive_data['type'] == 'bursty':
        net.add_bursty_drive(name=drive_data['name'],
                             tstart=drive_data['dynamics']['tstart'],
                             tstart_std=drive_data['dynamics']['tstart_std'],
                             tstop=drive_data['dynamics']['tstop'],
                             burst_rate=drive_data['dynamics']['burst_rate'],
                             burst_std=drive_data['dynamics']['burst_std'],
                             numspikes=drive_data['dynamics']['numspikes'],
                             spike_isi=drive_data['dynamics']['spike_isi'],
                             location=drive_data['location'],
                             n_drive_cells=_set_from_cell_specific(drive_data),
                             cell_specific=drive_data['cell_specific'],
                             weights_ampa=drive_data['weights_ampa'],
                             weights_nmda=drive_data['weights_nmda'],
                             synaptic_delays=drive_data['synaptic_delays'],
                             probability=drive_data["probability"],
                             event_seed=drive_data['event_seed'],
                             conn_seed=drive_data['conn_seed'])

    net.external_drives[drive_data['name']]['events'] = drive_data['events']
    if not read_output:
        net.external_drives[drive_data['name']]['events'] = list()


def _read_connectivity(net, conns_data):
    """Adds connections to a Network from json encoded connectivity"""
    # Overwrite drive connections
    net.connectivity = list()

    for conn_data in conns_data:
        src_gids = [int(s) for s in conn_data['gid_pairs'].keys()]
        target_gids_nested = [target_gid for target_gid
                              in conn_data['gid_pairs'].values()]
        conn_data['allow_autapses'] = bool(conn_data['allow_autapses'])
        net.add_connection(src_gids=src_gids,
                           target_gids=target_gids_nested,
                           loc=conn_data['loc'],
                           receptor=conn_data['receptor'],
                           weight=conn_data['nc_dict']['A_weight'],
                           delay=conn_data['nc_dict']['A_delay'],
                           lamtha=conn_data['nc_dict']['lamtha'],
                           allow_autapses=conn_data['allow_autapses'],
                           probability=conn_data['probability'])


def _read_rec_arrays(net, rec_arrays_data, read_output):
    """Adds rec arrays to Network from json data."""
    for key in rec_arrays_data:
        rec_array = rec_arrays_data[key]
        net.add_electrode_array(name=key,
                                electrode_pos=[
                                    tuple(pos) for
                                    pos in rec_array['positions']
                                ],
                                conductivity=rec_array['conductivity'],
                                method=rec_array['method'],
                                min_distance=rec_array['min_distance'])
        net.rec_arrays[key]._times = rec_array['times']
        net.rec_arrays[key]._data = rec_array['voltages']
        if not read_output:
            net.rec_arrays[key]._reset()


def _read_pos_dict(pos_dict):
    """Returns position dictionary with nested positions converted to tuple."""
    pos_dict_converted = dict()
    for key, value in pos_dict.items():
        if key == 'origin':
            pos_dict_converted[key] = tuple(value)
        else:
            pos_dict_converted[key] = [tuple(position) for position in value]
    return pos_dict_converted


def network_to_dict(net, write_output=False):
    """Returns a dict of parameters and outputs from Network.

    Parameters
    ----------
    net : Network
        hnn-core Network object
    write_output : bool
        Includes simulation-associated data.
    Returns
    -------
    dict
    """

    net_data = {
        'object_type': 'Network',
        'legacy_mode': net._legacy_mode,
        'N_pyr_x': net._N_pyr_x,
        'N_pyr_y': net._N_pyr_y,
        'celsius': net._params['celsius'],
        'cell_types': {name: template.to_dict()
                       for name, template in net.cell_types.items()
                       },
        'gid_ranges': {cell: {'start': c_range.start, 'stop': c_range.stop}
                       for cell, c_range in net.gid_ranges.items()
                       },
        'pos_dict': {cell: pos for cell, pos in net.pos_dict.items()},
        'cell_response': _cell_response_to_dict(net, write_output),
        'external_drives': {drive: _external_drive_to_dict(params,
                                                           write_output)
                            for drive, params in net.external_drives.items()
                            },
        'external_biases': net.external_biases,
        'connectivity': [_conn_to_dict(conn) for conn in net.connectivity],
        'rec_arrays': {ra_name: _rec_array_to_dict(ex_array, write_output)
                       for ra_name, ex_array in net.rec_arrays.items()
                       },
        'threshold': net.threshold,
        'delay': net.delay,
    }
    return net_data


@fill_doc
def write_network_configuration(net, fname, overwrite=True):
    """Writes network configuration to a json file.

    Writes network configurations as a hierarchical json similar to the Network
    object's structure. Outputs recorded during simulation such as currents and
    voltages are not saved due to size.

    Parameters
    ----------
    %(net)s
    %(fname)s
    %(overwrite)s

    Yields
    ------
    A json file containing the Network configurations.
    """

    if overwrite is False and os.path.exists(fname):
        raise FileExistsError('File already exists at path %s. Rename '
                              'the file or set overwrite=True.' % (fname,))

    net_data = net.to_dict(write_output=False)
    net_data_converted = _convert_np_array_to_list(net_data)

    # Saving file
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(net_data_converted, f, ensure_ascii=False, indent=4)


def _order_drives(gid_ranges, external_drives):
    """Returns an ordered dict of external drives by ascending gid ranges

    Drive data from hdf5 are ordered alphabetically by name. This function
    reorders the external drives by ascending gid ranges.

    Parameters
    ----------
    gid_ranges : dict (keys: names) of range
        Dictionary with connection or drive names as keys and ranges as values.
    external_drives: dict (keys: drive names) of dict (keys: parameters)
        Dictionary with drive name as keys and instances of _NetworkDrive as
        values.

    Returns
    -------
    OrderedDict : dict (keys: drive names) of dict (keys: parameters)
        Ordered dict with drives by ascending gid ranges
    """
    ordered_drives = OrderedDict()
    min_gid_to_drive = {min(gid_range): name
                        for (name, gid_range) in gid_ranges.items()
                        if name in external_drives.keys()
                        }
    min_gid_sorted = sorted(list(min_gid_to_drive.keys()))
    for min_gid in min_gid_sorted:
        drive_name = min_gid_to_drive[min_gid]
        ordered_drives[drive_name] = external_drives[drive_name]

    return ordered_drives


@fill_doc
def read_network_configuration(fname, read_drives=True):
    """Read network from a json configuration file.

    Parameters
    ----------
    %(fname)s
    %(read_drives)s

    Yields
    ------
    %(net)s
    """
    # Importing Network.
    # Cannot do this globally due to circular import.
    from .network import Network

    with open(fname, 'r') as file:
        net_data = json.load(file)

    if net_data.get('object_type') != 'Network':
        raise ValueError('The json should encode a Network object. '
                         'The file contains object of '
                         'type %s' % (net_data.get('object_type')))

    params = dict()
    params['celsius'] = net_data['celsius']
    params['threshold'] = net_data['threshold']

    mesh_shape = (net_data['N_pyr_x'], net_data['N_pyr_y'])

    # Instantiating network
    net = Network(params,
                  mesh_shape=mesh_shape,
                  legacy_mode=net_data['legacy_mode']
                  )

    # Setting attributes
    # Set cell types
    net.cell_types = _read_cell_types(net_data['cell_types'])
    # Set gid ranges
    gid_ranges_data = dict()
    for key in net_data['gid_ranges']:
        start = net_data['gid_ranges'][key]['start']
        stop = net_data['gid_ranges'][key]['stop']
        gid_ranges_data[key] = range(start, stop)
    net.gid_ranges = gid_ranges_data
    # Set pos_dict
    net.pos_dict = _read_pos_dict(net_data['pos_dict'])
    # Set cell_response
    net.cell_response = _read_cell_response(net_data['cell_response'],
                                            read_output=False)
    # Set external drives
    external_drive_data = _order_drives(net.gid_ranges,
                                        net_data['external_drives'])
    for key in external_drive_data.keys():
        _read_external_drive(net, external_drive_data[key],
                             read_output=False, read_drives=read_drives)
    # Set external biases
    net.external_biases = net_data['external_biases']
    # Set connectivity
    _read_connectivity(net, net_data['connectivity'])
    # Set rec_arrays
    _read_rec_arrays(net, net_data['rec_arrays'], read_output=False)
    # Set threshold
    net.threshold = net_data['threshold']
    # Set delay
    net.delay = net_data['delay']

    return net
