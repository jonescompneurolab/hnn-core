"""Network io"""

# Authors: Rajat Partani <rajatpartani@gmail.com>
#          Mainak Jas <mjas@mgh.harvard.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>

import os
from h5io import write_hdf5, read_hdf5

from .cell import Cell, Section
from .cell_response import CellResponse
from .docs import fill_doc


@fill_doc
def write_network(net, fname, overwrite=True, save_unsimulated=False):
    """Write network to a file.

    Parameters
    ----------
    %(net)s
    %(fname)s
    %(overwrite)s
    %(save_unsimulated)s

    Outputs
    -------
    A hdf5 file containing the Network object

    File Content Description
    ------------------------
    %(network_file_content_description)s

    cell type description
    ---------------------
    %(cell_description)s

    gid range description
    ---------------------
    %(gid_range_description)s

    external drive description
    --------------------------
    %(external_drive_description)s

    external bias description
    -------------------------
    %(external_bias_description)s

    connection description
    ----------------------
    %(connection_description)s

    extracellular array description
    -------------------------------
    %(extracellular_array_description)s

    """
    if overwrite is False and os.path.exists(fname):
        raise FileExistsError('File already exists at path %s. Rename '
                              'the file or set overwrite=True.' % (fname,))
    net_data = dict()
    net_data['object_type'] = "Network"
    net_data['N_pyr_x'] = net._N_pyr_x
    net_data['N_pyr_y'] = net._N_pyr_y
    net_data['celsius'] = net._params['celsius']
    cell_types_data = dict()
    for key in net.cell_types:
        cell_copy = net.cell_types[key].copy()
        cell_copy.build()
        cell_types_data[key] = cell_copy.to_dict()
    net_data['cell_types'] = cell_types_data
    # Write gid_ranges
    gid_ranges_data = dict()
    for key in net.gid_ranges:
        gid_ranges_data[key] = dict()
        gid_ranges_data[key]['start'] = net.gid_ranges[key].start
        gid_ranges_data[key]['stop'] = net.gid_ranges[key].stop
    net_data['gid_ranges'] = gid_ranges_data
    # Write pos_dict
    pos_dict_data = dict()
    for key in net.pos_dict:
        pos_dict_data[key] = net.pos_dict[key]
    net_data['pos_dict'] = pos_dict_data
    # Write cell_response
    if (not net.cell_response) or save_unsimulated:
        net_data['cell_response'] = None
    else:
        net_data['cell_response'] = net.cell_response.to_dict()
    # Write External drives
    external_drives_data = dict()
    for key in net.external_drives.keys():
        external_drives_data[key] = (_external_drive_to_dict
                                     (net.external_drives[key],
                                      save_unsimulated))
    net_data['external_drives'] = external_drives_data
    # Write External biases
    net_data['external_biases'] = net.external_biases
    # Write connectivity
    net_data['connectivity'] = _write_connectivity(net.connectivity)
    # Write rec arrays
    net_data['rec_arrays'] = dict()
    for key in net.rec_arrays.keys():
        rec_array_copy = net.rec_arrays[key].copy()
        if save_unsimulated:
            rec_array_copy._reset()
        net_data['rec_arrays'][key] = rec_array_copy.to_dict()
    # Write threshold
    net_data['threshold'] = net.threshold
    # Write delay
    net_data['delay'] = net.delay

    # Saving file
    write_hdf5(fname, net_data, overwrite=overwrite)


@fill_doc
def read_network(fname, read_raw=False):
    """Read network from a file.

    Parameters
    ----------
    %(fname)s
    %(read_raw)s

    Outputs
    -------
    %(net)s

    File Content Description
    ------------------------
    %(network_file_content_description)s

    cell type description
    ---------------------
    %(cell_description)s

    gid range description
    ---------------------
    %(gid_range_description)s

    external drive description
    --------------------------
    %(external_drive_description)s

    external bias description
    -------------------------
    %(external_bias_description)s

    connection description
    ----------------------
    %(connection_description)s

    extracellular array description
    -------------------------------
    %(extracellular_array_description)s
    """
    net_data = read_hdf5(fname)
    if 'object_type' not in net_data:
        raise NameError('The given file is not compatible. '
                        'The file should contain information'
                        ' about object type to be read.')
    if net_data['object_type'] != 'Network':
        raise ValueError('The object should be of type Network. '
                         'The file contains object of '
                         'type %s' % (net_data['object_type'],))
    params = dict()
    params['N_pyr_x'] = net_data['N_pyr_x']
    params['N_pyr_y'] = net_data['N_pyr_y']
    params['celsius'] = net_data['celsius']
    params['threshold'] = net_data['threshold']

    # Instantiating network
    # Cannot do this globally due to circular import
    from .network import Network
    net = Network(params)

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
                                            read_raw)
    # Set external drives
    for key in net_data['external_drives'].keys():
        _read_external_drive(net, net_data['external_drives'][key], read_raw)
    # Set external biases
    net.external_biases = net_data['external_biases']
    # Set connectivity
    _read_connectivity(net, net_data['connectivity'])
    # Set rec_arrays
    _read_rec_arrays(net, net_data['rec_arrays'], read_raw)
    # Set threshold
    net.threshold = net_data['threshold']
    # Set delay
    net.delay = net_data['delay']

    return net


def _write_connectivity(connectivity):
    conns_data = list()
    for conn in connectivity:
        conn_data = dict()
        conn_data['target_type'] = conn['target_type']
        conn_data['target_gids'] = list(conn['target_gids'])
        conn_data['num_targets'] = conn['num_targets']
        conn_data['src_type'] = conn['src_type']
        conn_data['src_gids'] = list(conn['src_gids'])
        conn_data['num_srcs'] = conn['num_srcs']
        gid_pairs = (dict((str(key), val)
                     for key, val in conn['gid_pairs'].items()))
        conn_data['gid_pairs'] = gid_pairs
        conn_data['loc'] = conn['loc']
        conn_data['receptor'] = conn['receptor']
        conn_data['nc_dict'] = conn['nc_dict']
        conn_data['allow_autapses'] = int(conn['allow_autapses'])
        conn_data['probability'] = conn['probability']
        conns_data.append(conn_data)
    return conns_data


def _external_drive_to_dict(drive, save_unsimulated):
    drive_data = dict()
    for key in drive.keys():
        # Cannot store sets with hdf5
        if isinstance(drive[key], set):
            drive_data[key] = list(drive[key])
        else:
            drive_data[key] = drive[key]
    if save_unsimulated:
        drive_data['events'] = list()
    return drive_data


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
        cell_types[cell_name] = Cell(name=cell_data['name'],
                                     pos=cell_data['pos'],
                                     sections=sections,
                                     synapses=cell_data['synapses'],
                                     topology=cell_data['topology'],
                                     sect_loc=cell_data['sect_loc'],
                                     gid=cell_data['gid'])
        # Setting cell attributes
        cell_types[cell_name].dipole_pp = cell_data['dipole_pp']
        cell_types[cell_name].vsec = cell_data['vsec']
        cell_types[cell_name].isec = cell_data['isec']
        cell_types[cell_name].tonic_biases = cell_data['tonic_biases']

    return cell_types


def _read_cell_response(cell_response_data, read_raw):
    if (not cell_response_data) or read_raw:
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


def _read_external_drive(net, drive_data, read_raw):
    if drive_data['type'] == 'evoked':
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
                             event_seed=drive_data['event_seed'],
                             conn_seed=drive_data['conn_seed'])
    elif drive_data['type'] == 'poisson':
        net.add_poisson_drive(name=drive_data['name'],
                              tstart=drive_data['dynamics']['tstart'],
                              tstop=drive_data['dynamics']['tstop'],
                              rate_constant=(drive_data['dynamics']
                                                       ['rate_constant']),
                              location=drive_data['location'],
                              n_drive_cells=drive_data['n_drive_cells'],
                              cell_specific=drive_data['cell_specific'],
                              weights_ampa=drive_data['weights_ampa'],
                              weights_nmda=drive_data['weights_nmda'],
                              synaptic_delays=drive_data['synaptic_delays'],
                              event_seed=drive_data['event_seed'],
                              conn_seed=drive_data['conn_seed'])
    elif drive_data['type'] == 'bursty':
        net.add_bursty_drive(name=drive_data['name'],
                             tstart=drive_data['dynamics']['tstart'],
                             tstart_std=drive_data['dynamics']['tstart_std'],
                             tstop=drive_data['dynamics']['tstop'],
                             burst_rate=drive_data['dynamics']['burst_rate'],
                             burst_std=drive_data['dynamics']['burst_std'],
                             num_spikes=drive_data['dynamics']['num_spikes'],
                             spike_isi=drive_data['dynamics']['spike_isi'],
                             location=drive_data['location'],
                             n_drive_cells=drive_data['n_drive_cells'],
                             cell_specific=drive_data['cell_specific'],
                             weights_ampa=drive_data['weights_ampa'],
                             weights_nmda=drive_data['weights_nmda'],
                             synaptic_delays=drive_data['synaptic_delays'],
                             event_seed=drive_data['event_seed'],
                             conn_seed=drive_data['conn_seed'])

    net.external_drives[drive_data['name']]['events'] = drive_data['events']
    if read_raw:
        net.external_drives[drive_data['name']]['events'] = list()


def _read_connectivity(net, conns_data):
    # Overwrite drive connections
    net.connectivity = list()
    for conn_data in conns_data:
        conn_data['allow_autapses'] = bool(conn_data['allow_autapses'])
        # conn_data['allow_autapses'] = bool(conn_data['allow_autapses'])
        net.add_connection(src_gids=conn_data['src_type'],
                           target_gids=conn_data['target_type'],
                           loc=conn_data['loc'],
                           receptor=conn_data['receptor'],
                           weight=conn_data['nc_dict']['A_weight'],
                           delay=conn_data['nc_dict']['A_delay'],
                           lamtha=conn_data['nc_dict']['lamtha'],
                           allow_autapses=conn_data['allow_autapses'],
                           probability=conn_data['probability'])


def _read_rec_arrays(net, rec_arrays_data, read_raw):
    for key in rec_arrays_data:
        rec_array = rec_arrays_data[key]
        net.add_electrode_array(name=key,
                                electrode_pos=rec_array['positions'],
                                conductivity=rec_array['conductivity'],
                                method=rec_array['method'],
                                min_distance=rec_array['min_distance'])
        net.rec_arrays[key]._times = rec_array['times']
        net.rec_arrays[key]._data = rec_array['voltages']
        if read_raw:
            net.rec_arrays[key]._reset()
