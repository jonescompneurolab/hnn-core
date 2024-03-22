"""Network io"""

# Authors: Rajat Partani <rajatpartani@gmail.com>
#          Mainak Jas <mjas@mgh.harvard.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          George Dang <george_dang@brown.edu>

import os
from h5io import write_hdf5, read_hdf5

from .cell import Cell, Section
from .cell_response import CellResponse
from .externals.mne import fill_doc


def _cell_response_to_dict(net, write_output):
    # Write cell_response as dict
    if (not net.cell_response) or (not write_output):
        return None
    else:
        return net.cell_response.to_dict()


def _rec_array_to_dict(value, write_output):
    rec_array_copy = value.copy()
    if not write_output:
        rec_array_copy._reset()
    rec_array_copy_dict = rec_array_copy.to_dict()
    return rec_array_copy_dict


def _connectivity_to_list_of_dicts(connectivity):

    def _conn_to_dict(conn):
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

    conns_data = [_conn_to_dict(conn) for conn in connectivity]

    return conns_data


def _external_drive_to_dict(drive, write_output):
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
    node_tuple = node_string.split(',')
    node_tuple[1] = int(node_tuple[1])
    node = (node_tuple[0], node_tuple[1])
    return node


def _read_cell_types(cell_types_data):
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
                                     pos=cell_data['pos'],
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


def _read_external_drive(net, drive_data, read_output, read_drives):
    if not read_drives:
        return None

    def _set_from_cell_specific(drive_data):
        """Returns number of drive cells based on cell_specific bool

        The n_drive_cells keyword for add_poisson_drive and add_bursty_drive
        methods accept either an int or string (n_cells). If the bool keyword
        cell_specific = True, n_drive_cells must be 'n_cells'.
        """
        if drive_data['cell_specific']:
            return 'n_cells'
        return drive_data['n_drive_cells']

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
    # Overwrite drive connections
    net.connectivity = list()

    for _i, conn_data in enumerate(conns_data):
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
    for key in rec_arrays_data:
        rec_array = rec_arrays_data[key]
        net.add_electrode_array(name=key,
                                electrode_pos=rec_array['positions'],
                                conductivity=rec_array['conductivity'],
                                method=rec_array['method'],
                                min_distance=rec_array['min_distance'])
        net.rec_arrays[key]._times = rec_array['times']
        net.rec_arrays[key]._data = rec_array['voltages']
        if not read_output:
            net.rec_arrays[key]._reset()


@fill_doc
def write_network(net, fname, overwrite=True, write_output=True, source='obj'):
    """Write network to a HDF5 file.

    Parameters
    ----------
    %(net)s
    %(fname)s
    %(overwrite)s
    %(write_output)s
    %(source)s

    Yields
    ------
    A hdf5 file containing the Network object.
    """
    if overwrite is False and os.path.exists(fname):
        raise FileExistsError('File already exists at path %s. Rename '
                              'the file or set overwrite=True.' % (fname,))

    net_data = {
        'object_type': 'Network',
        'source': source,
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
        'connectivity': _connectivity_to_list_of_dicts(net.connectivity),
        'rec_arrays': {ra_name: _rec_array_to_dict(ex_array, write_output)
                       for ra_name, ex_array in net.rec_arrays.items()
                       },
        'threshold': net.threshold,
        'delay': net.delay,
    }

    # Saving file
    write_hdf5(fname, net_data, overwrite=overwrite)


@fill_doc
def read_network(fname, read_output=True, read_drives=True):
    """Read network from a file.

    Parameters
    ----------
    %(fname)s
    %(read_output)s

    Yields
    ------
    %(net)s
    """
    # Importing Network.
    # Cannot do this globally due to circular import.
    from .network import Network
    net_data = read_hdf5(fname)
    if 'object_type' not in net_data:
        raise NameError('The given file is not compatible. '
                        'The file should contain information'
                        ' about object type to be read.')
    if net_data['object_type'] != 'Network':
        raise ValueError('The object should be of type Network. '
                         'The file contains object of '
                         'type %s' % (net_data['object_type'],))
    legacy_mode = False
    if net_data['source'] == 'param':
        legacy_mode = True

    params = dict()
    params['celsius'] = net_data['celsius']
    params['threshold'] = net_data['threshold']

    mesh_shape = (net_data['N_pyr_x'], net_data['N_pyr_y'])

    # Instantiating network
    net = Network(params, mesh_shape=mesh_shape, legacy_mode=legacy_mode)

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
    net.pos_dict = net_data['pos_dict']
    # Set cell_response
    net.cell_response = _read_cell_response(net_data['cell_response'],
                                            read_output)
    # Set external drives
    for key in net_data['external_drives'].keys():
        _read_external_drive(net, net_data['external_drives'][key],
                             read_output, read_drives)
    # Set external biases
    net.external_biases = net_data['external_biases']
    # Set connectivity
    _read_connectivity(net, net_data['connectivity'])
    # Set rec_arrays
    _read_rec_arrays(net, net_data['rec_arrays'], read_output)
    # Set threshold
    net.threshold = net_data['threshold']
    # Set delay
    net.delay = net_data['delay']

    return net
