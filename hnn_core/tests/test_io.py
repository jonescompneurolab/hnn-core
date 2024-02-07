from pathlib import Path
from numpy.testing import assert_allclose
from h5io import write_hdf5, read_hdf5
import pytest
import numpy as np

from hnn_core import (read_network, simulate_dipole, read_params,
                      jones_2009_model, calcium_model,
                      )

from hnn_core.hnn_io import (_cell_response_to_dict, _rec_array_to_dict,
                             _connectivity_to_list_of_dicts,
                             _external_drive_to_dict, _str_to_node
                             )

hnn_core_root = Path(__file__).parents[1]


@pytest.fixture
def params():
    params_path = Path(hnn_core_root, 'param', 'default.json')
    params = read_params(params_path)
    params['celsius'] = 37.0
    params['threshold'] = 0.0

    return params


@pytest.fixture
def jones_2009_network(params):

    # Instantiating network along with drives
    net = jones_2009_model(params=params, add_drives_from_params=True,
                           mesh_shape=(3, 3))

    # Adding bias
    net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)

    # Adding electrode arrays
    electrode_pos = (1, 2, 3)
    net.add_electrode_array('el1', electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array('arr1', electrode_pos)

    return net


@pytest.fixture
def calcium_network(params):
    # Instantiating network along with drives
    net = calcium_model(params=params, add_drives_from_params=True,
                        mesh_shape=(3, 3))

    # Adding bias
    net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)

    # Adding electrode arrays
    electrode_pos = (1, 2, 3)
    net.add_electrode_array('el1', electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array('arr1', electrode_pos)

    return net


def test_eq(jones_2009_network, calcium_network):
    net1 = jones_2009_network
    net2 = calcium_network

    # Check eq of same network
    assert net1 == net1
    # Check eq of different networks
    assert not net1 == net2

    # Check a change in connectivity
    net1_clear_conn = net1.copy()
    net1_clear_conn.clear_connectivity()
    assert net1_clear_conn != net1

    # Hardwired change in connectivity attribute
    net1_hard_change_conn = net1.copy()
    net1_hard_change_conn.connectivity[0]['gid_pairs'] = {}
    assert net1_hard_change_conn != net1

    # Check change in drives
    net1_clear_drive = net1.copy()
    net1_clear_drive.clear_drives()
    assert net1_clear_drive != net1

    # Hardwired change in drive attribute
    net1_hard_change_drive = net1.copy()
    net1_hard_change_drive.external_drives['type'] = ''
    assert net1_hard_change_drive != net1


def test_write_network(tmp_path, jones_2009_network):
    """ Tests that a hdf5 file is written """

    # Check no file is already written
    path_out = tmp_path / 'net.hdf5'
    assert not path_out.is_file()

    # Write network check
    jones_2009_network.write(path_out)
    assert path_out.is_file()

    # Overwrite network check
    last_mod_time1 = path_out.stat().st_mtime
    jones_2009_network.write(path_out)
    last_mod_time2 = path_out.stat().st_mtime
    assert last_mod_time1 < last_mod_time2

    # No overwrite check
    with pytest.raises(FileExistsError,
                       match="File already exists at path "):
        jones_2009_network.write(path_out, overwrite=False)


def test_write_network_no_output(tmp_path, jones_2009_network):
    net = jones_2009_network.copy()
    path_out = tmp_path / 'net.hdf5'

    # Simulate and check for output
    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    assert np.array_equal(net.rec_arrays['el1'].times,
                          [0.0, 0.5, 1.0, 1.5, 2.0]
                          )

    # Write to file
    net.write(path_out, write_output=False)

    # Read in file and check no outputs were written
    hdf5_read = read_hdf5(path_out)
    assert not any([bool(val['times'])
                    for val in hdf5_read['rec_arrays'].values()]
                   )
    assert not any([bool(val['voltages'])
                    for val in hdf5_read['rec_arrays'].values()]
                   )
    assert not any([bool(val['events'])
                    for val in hdf5_read['external_drives'].values()]
                   )


def test_cell_response_to_dict(jones_2009_network):
    net = jones_2009_network

    # No simulation so should have None for cell response
    result1 = _cell_response_to_dict(net, write_output=True)
    assert result1 is None

    # Check for cell response dictionary after a simulation
    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    assert net.cell_response is not None
    result2 = _cell_response_to_dict(net, write_output=True)
    assert bool(result2) and isinstance(result2, dict)

    # Check for None if kw supplied
    result3 = _cell_response_to_dict(net, write_output=False)
    assert result3 is None


def test_rec_array_to_dict(jones_2009_network):
    net = jones_2009_network

    # Check rec array times and voltages are in dict after simulation
    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    result = _rec_array_to_dict(net.rec_arrays['el1'], write_output=True)
    assert isinstance(result, dict)
    assert all([key in result for key in ['positions', 'conductivity',
                                          'method', 'min_distance',
                                          'times', 'voltages'
                                          ]
                ]
               )
    assert np.array_equal(result['times'], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert result['voltages'].shape == (1, 1, 5)

    # Check values are empty if write_output keyword is false
    result2 = _rec_array_to_dict(net.rec_arrays['el1'], write_output=False)
    assert result2['times'].size == 0
    assert result2['voltages'].size == 0


def test_connectivity_to_list_of_dicts(jones_2009_network):
    net = jones_2009_network

    result = _connectivity_to_list_of_dicts(net.connectivity[0:2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {'target_type': 'L2_basket',
                         'target_gids': [0, 1, 2],
                         'num_targets': 3,
                         'src_type': 'evdist1',
                         'src_gids': [24, 25, 26],
                         'num_srcs': 3,
                         'gid_pairs': {'24': [0], '25': [1], '26': [2]},
                         'loc': 'distal',
                         'receptor': 'ampa',
                         'nc_dict': {'A_delay': 0.1,
                                     'A_weight': 0.006562,
                                     'lamtha': 3.0,
                                     'threshold': 0.0},
                         'allow_autapses': 1,
                         'probability': 1.0}


def test_external_drive_to_dict(jones_2009_network):
    net = jones_2009_network

    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    first_key = list(net.external_drives.keys())[0]
    result = _external_drive_to_dict(net.external_drives[first_key],
                                     write_output=True
                                     )
    assert isinstance(result, dict)
    assert all([key in result for key in ['type', 'location', 'n_drive_cells',
                                          'event_seed', 'conn_seed',
                                          'dynamics', 'events', 'weights_ampa',
                                          'weights_nmda', 'synaptic_delays',
                                          'probability', 'name',
                                          'target_types', 'cell_specific'
                                          ]
                ]
               )
    assert len(result['events'][0]) == 21

    result2 = _external_drive_to_dict(net.external_drives[first_key],
                                      write_output=False
                                      )
    assert len(result2['events']) == 0


def test_str_to_node():
    """ Creates a tuple (str,int) from string with a comma """
    result = _str_to_node('cell_name,0')
    assert isinstance(result, tuple)
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)


def test_read_hdf5(jones_2009_network):

    # jones_2009_network.write(Path('.', 'assets/jones2009_test_read.hdf5'))
    # This file is written from the jones_2009_network
    net = read_network(Path('assets', 'jones2009_test_read.hdf5'))
    assert net == jones_2009_network


def test_read_hdf5_with_simulation(jones_2009_network):
    # Test reading a network with simulation
    net_sim = read_network(
        Path('assets', 'jones2009_simple_sim_test_read.hdf5')
    )
    assert net_sim.rec_arrays['el1'].voltages.size != 0
    assert len(net_sim.external_drives['evdist1']['events']) > 0

    # Test reading file without simulation information
    net_sim_output_false = read_network(
        Path('assets', 'jones2009_simple_sim_test_read.hdf5'),
        read_output=False
    )
    assert net_sim_output_false.rec_arrays['el1'].voltages.size == 0
    assert len(net_sim_output_false.external_drives['evdist1']['events']) == 0

    # Test reading file with simulation and without drive information
    net_sim_drives_false = read_network(
        Path('assets', 'jones2009_simple_sim_test_read.hdf5'),
        read_output=True,
        read_drives=False
    )
    assert net_sim_drives_false.rec_arrays['el1'].voltages.size != 0
    assert not bool(net_sim_drives_false.external_drives)

    # Test reading file without simulation and drive information
    net_sim_output_false_drives_false = read_network(
        Path('assets', 'jones2009_simple_sim_test_read.hdf5'),
        read_output=False,
        read_drives=False
    )
    assert (net_sim_output_false_drives_false
            .rec_arrays['el1'].voltages.size == 0)
    assert not bool(net_sim_output_false_drives_false.external_drives)


def test_read_incorrect_format(tmp_path):

    # Checking object type field not exists error
    dummy_data = dict()
    dummy_data['objective'] = "Check Object type errors"
    write_hdf5(tmp_path / 'not_net.hdf5', dummy_data, overwrite=True)
    with pytest.raises(NameError,
                       match="The given file is not compatible."):
        read_network(tmp_path / 'not_net.hdf5')

    # Checking wrong object type error
    dummy_data['object_type'] = "net"
    write_hdf5(tmp_path / 'not_net.hdf5', dummy_data, overwrite=True)
    with pytest.raises(ValueError,
                       match="The object should be of type Network."):
        read_network(tmp_path / 'not_net.hdf5')


def test_simulate_from_read(jones_2009_network):
    """
    Tests a simulation from a read-in network creates a similar simulation to
    the reference network the input file was created from.
    """
    net = jones_2009_network
    dpls1 = simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)

    net_read = read_network(Path('assets', 'jones2009_test_read.hdf5'))
    dpls2 = simulate_dipole(net_read, tstop=2, n_trials=1, dt=0.5)

    for dpl1, dpl2 in zip(dpls1, dpls2):
        assert_allclose(dpl1.times, dpl2.times, rtol=0.00051, atol=0)
        for dpl_key in dpl1.data.keys():
            assert_allclose(dpl1.data[dpl_key],
                            dpl2.data[dpl_key], rtol=0.000051, atol=0)

    # Smoke test
    net_read.plot_cells(show=False)
