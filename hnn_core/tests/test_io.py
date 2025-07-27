# Authors: George Dang <george_dang@brown.edu>
#          Rajat Partani <rajatpartani@gmail.com>

from pathlib import Path
from time import sleep
import pytest
import numpy as np
import json

from hnn_core import (
    simulate_dipole,
    read_params,
    calcium_model,
)

from hnn_core.hnn_io import (
    _cell_response_to_dict,
    _rec_array_to_dict,
    _external_drive_to_dict,
    _str_to_node,
    _conn_to_dict,
    _order_drives,
    read_network_configuration,
)

from regenerate_test_network import jones_2009_additional_features

hnn_core_root = Path(__file__).parents[1]
assets_path = Path(hnn_core_root, "tests", "assets")


@pytest.fixture
def params():
    params_path = Path(hnn_core_root, "param", "default.json")
    params = read_params(params_path)
    params["celsius"] = 37.0
    params["threshold"] = 0.0

    return params


@pytest.fixture
def jones_2009_network():
    # This allows us to define this test network once, but use it as both a
    # fixture here in this file, or regenerate the network itself if used
    # elsewhere.
    net = jones_2009_additional_features()

    return net


@pytest.fixture
def calcium_network(params):
    # Instantiating network along with drives
    net = calcium_model(params=params, add_drives_from_params=True, mesh_shape=(3, 3))

    # Adding bias
    tonic_bias = {"L2_pyramidal": 1.0}
    net.add_tonic_bias(amplitude=tonic_bias)

    # Adding electrode arrays
    electrode_pos = (1, 2, 3)
    net.add_electrode_array("el1", electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array("arr1", electrode_pos)

    return net


def test_eq(jones_2009_network, calcium_network):
    net1 = jones_2009_network
    net2 = calcium_network

    # Check eq of same network
    assert net1 == net1
    # Check eq of different networks
    assert not net1 == net2

    # Check change in drives
    net1_clear_drive = net1.copy()
    net1_clear_drive.clear_drives()
    assert net1_clear_drive != net1

    # Hardwired change in drive attribute
    net1_hard_change_drive = net1.copy()
    net1_hard_change_drive.external_drives["type"] = ""
    assert net1_hard_change_drive != net1

    # Hardwired change in drive weights
    net1_hard_change_drive = net1.copy()
    (net1_hard_change_drive.external_drives["evdist1"]["weights_ampa"]["L2_basket"]) = 0
    assert net1_hard_change_drive != net1


def test_eq_conn(jones_2009_network):
    net1 = jones_2009_network

    # Check a change in connectivity
    net1_clear_conn = net1.copy()
    net1_clear_conn.clear_connectivity()
    assert net1_clear_conn != net1

    # Hardwired change in connectivity attribute
    net1_hard_change_conn = net1.copy()
    net1_hard_change_conn.connectivity[0]["gid_pairs"] = {}
    assert net1_hard_change_conn != net1

    # Hardwired change in connectivity nc_dict
    net1_hard_change_conn = net1.copy()
    net1_hard_change_conn.connectivity[0]["nc_dict"]["A_weight"] = 0
    assert net1_hard_change_conn != net1

    # Check edge case, same number of connections, different replicate in conn
    net1_alt_conn1 = net1.copy()
    net1_alt_conn2 = net1.copy()
    l_conn = net1_alt_conn1.connectivity
    l_conn_rep_start = [l_conn[0]] + l_conn
    l_conn_rep_end = l_conn + [l_conn[-1]]
    net1_alt_conn1.connectivity = l_conn_rep_start
    net1_alt_conn2.connectivity = l_conn_rep_end
    assert net1 != net1_alt_conn1
    assert net1_alt_conn1 == net1_alt_conn1
    assert net1_alt_conn1 != net1_alt_conn2


def test_write_configuration(tmp_path, jones_2009_network):
    """Tests that a json file is written"""

    net = jones_2009_network.copy()
    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)

    # Check no file is already written
    path_out = tmp_path / "net.json"
    assert not path_out.is_file()

    # Write network check
    jones_2009_network.write_configuration(path_out)
    assert path_out.is_file()

    # Overwrite network check
    last_mod_time1 = path_out.stat().st_mtime
    sleep(0.05)
    jones_2009_network.write_configuration(path_out)
    last_mod_time2 = path_out.stat().st_mtime
    assert last_mod_time1 < last_mod_time2

    # No overwrite check
    with pytest.raises(FileExistsError, match="File already exists at path "):
        jones_2009_network.write_configuration(path_out, overwrite=False)

    # Check no outputs were written
    with open(path_out) as file:
        read_in = json.load(file)

    assert not any([bool(val["times"]) for val in read_in["rec_arrays"].values()])
    assert not any([bool(val["voltages"]) for val in read_in["rec_arrays"].values()])
    assert not any([bool(val["events"]) for val in read_in["external_drives"].values()])
    assert read_in["cell_response"] == {}


def test_cell_response_to_dict(jones_2009_network):
    """Tests _cell_response_to_dict function"""
    net = jones_2009_network

    # When a simulation hasn't been run, return an empty dict
    result1 = _cell_response_to_dict(net, write_output=True)
    assert result1 == dict()

    # Check for cell response dictionary after a simulation
    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    assert net.cell_response is not None
    result2 = _cell_response_to_dict(net, write_output=True)
    assert bool(result2) and isinstance(result2, dict)

    # Check for empty dict if kw supplied
    result3 = _cell_response_to_dict(net, write_output=False)
    assert result3 == dict()


def test_rec_array_to_dict(jones_2009_network):
    """Tests _rec_array_to_dict function"""
    net = jones_2009_network

    # Check rec array times and voltages are in dict after simulation
    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    result = _rec_array_to_dict(net.rec_arrays["el1"], write_output=True)
    assert isinstance(result, dict)
    assert all(
        [
            key in result
            for key in [
                "positions",
                "conductivity",
                "method",
                "min_distance",
                "times",
                "voltages",
            ]
        ]
    )
    assert np.array_equal(result["times"], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert result["voltages"].shape == (1, 1, 5)

    # Check values are empty if write_output keyword is false
    result2 = _rec_array_to_dict(net.rec_arrays["el1"], write_output=False)
    assert result2["times"].size == 0
    assert result2["voltages"].size == 0


def test_conn_to_dict(jones_2009_network):
    """Tests _connectivity_to_list_of_dicts function"""
    net = jones_2009_network

    result = _conn_to_dict(net.connectivity[0])
    assert isinstance(result, dict)
    assert result == {
        "target_type": "L2_basket",
        "target_gids": [0, 1, 2],
        "num_targets": 3,
        "src_type": "evdist1",
        "src_gids": [24, 25, 26],
        "num_srcs": 3,
        "gid_pairs": {"24": [0], "25": [1], "26": [2]},
        "loc": "distal",
        "receptor": "ampa",
        "nc_dict": {
            "A_delay": 0.1,
            "A_weight": 0.006562,
            "lamtha": 3.0,
            "threshold": 0.0,
            "gain": 1.0,
        },
        "allow_autapses": 1,
        "probability": 1.0,
    }


def test_external_drive_to_dict(jones_2009_network):
    """Tests _external_drive_to_dict function"""
    net = jones_2009_network

    simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    first_key = list(net.external_drives.keys())[0]
    result = _external_drive_to_dict(net.external_drives[first_key], write_output=True)
    assert isinstance(result, dict)
    assert all(
        [
            key in result
            for key in [
                "type",
                "location",
                "n_drive_cells",
                "event_seed",
                "conn_seed",
                "dynamics",
                "events",
                "weights_ampa",
                "weights_nmda",
                "synaptic_delays",
                "probability",
                "name",
                "target_types",
                "cell_specific",
            ]
        ]
    )
    assert len(result["events"][0]) == 21

    result2 = _external_drive_to_dict(
        net.external_drives[first_key], write_output=False
    )
    assert len(result2["events"]) == 0


def test_str_to_node():
    """Creates a tuple (str,int) from string with a comma"""
    result = _str_to_node("cell_name,0")
    assert isinstance(result, tuple)
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)


def test_order_drives(jones_2009_network):
    """Reorders drive dict by ascending range order"""
    drive_names = list(jones_2009_network.external_drives.keys())
    drive_names_alpha = sorted(drive_names)
    drives_reordered = {
        name: jones_2009_network.external_drives for name in drive_names_alpha
    }
    assert list(drives_reordered.keys()) == [
        "alpha_prox",
        "evdist1",
        "evprox1",
        "evprox2",
        "poisson",
    ]

    drives_by_range = _order_drives(jones_2009_network.gid_ranges, drives_reordered)
    assert list(drives_by_range.keys()) == [
        "evdist1",
        "evprox1",
        "evprox2",
        "alpha_prox",
        "poisson",
    ]


def test_read_configuration_json(jones_2009_network):
    """Read-in of a hdf5 file"""
    net = read_network_configuration(Path(assets_path, "jones2009_3x3_drives.json"))
    assert net == jones_2009_network

    # Read without drives
    net_no_drives = read_network_configuration(
        Path(assets_path, "jones2009_3x3_drives.json"), read_drives=False
    )
    # Check there are no external drives
    assert len(net_no_drives.external_drives) == 0
    # Check there are no external drive connections
    connection_src_types = [
        connection["src_type"] for connection in net_no_drives.connectivity
    ]
    assert not any(
        [src_type in net.external_drives.keys() for src_type in connection_src_types]
    )

    # Read without external bias
    net_no_bias = read_network_configuration(
        Path(assets_path, "jones2009_3x3_drives.json"), read_external_biases=False
    )
    assert len(net_no_bias.external_biases) == 0
    assert len(net_no_bias.external_drives) > 0


def test_read_incorrect_format(tmp_path):
    """Test that error raise when the json do not have a Network label."""

    # Checking object type field not exists error
    dummy_data = dict()

    dummy_data["object_type"] = "NotNetwork"
    file_path = tmp_path / "not_net.json"
    with open(file_path, "w") as file:
        json.dump(dummy_data, file)

    with pytest.raises(ValueError, match="The json should encode a Network object."):
        read_network_configuration(file_path)
