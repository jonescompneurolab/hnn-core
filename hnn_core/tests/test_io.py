import os.path as op
from numpy.testing import assert_allclose
from h5io import write_hdf5
import pytest

import hnn_core
from hnn_core import read_network
from hnn_core import simulate_dipole
from hnn_core import read_params
from hnn_core import jones_2009_model, law_2021_model, calcium_model

hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')


@pytest.mark.parametrize("network_model",
                         [law_2021_model, calcium_model,
                          jones_2009_model])
def test_network_io(tmp_path, network_model):
    # For simulation to be shorter(Discuss)
    params = op.join(hnn_core_root, 'param', 'default.json')
    if isinstance(params, str):
        params = read_params(params)
    params['N_pyr_x'] = 3
    params['N_pyr_y'] = 3
    params['celsius'] = 37.0
    params['threshold'] = 0.0

    # Instantiating network along with drives
    net = network_model(params=params, add_drives_from_params=True)

    # Adding bias
    net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)

    # Test __eq__ method
    net_copy = net.copy()
    assert net_copy == net

    # Adding electrode arrays
    electrode_pos = (1, 2, 3)
    net.add_electrode_array('el1', electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array('arr1', electrode_pos)

    # Writing network
    net.write(tmp_path / 'net.hdf5')

    # Testing when overwrite is False and same filename is used
    with pytest.raises(FileExistsError,
                       match="File already exists at path "):
        net.write(tmp_path / 'net.hdf5', overwrite=False)

    # Reading network
    net_read = read_network(tmp_path / 'net.hdf5')
    assert net == net_read

    # Simulating network
    dpls1 = simulate_dipole(net, tstop=2, n_trials=1, dt=0.5)
    dpls2 = simulate_dipole(net_read, tstop=2, n_trials=1, dt=0.5)
    for dpl1, dpl2 in zip(dpls1, dpls2):
        assert_allclose(dpl1.times, dpl2.times, rtol=0.00051, atol=0)
        for dpl_key in dpl1.data.keys():
            assert_allclose(dpl1.data[dpl_key],
                            dpl2.data[dpl_key], rtol=0.000051, atol=0)
    # Writing simulated network and reading it
    net.write(tmp_path / 'net_sim.hdf5')
    net_sim = read_network(tmp_path / 'net_sim.hdf5')
    assert net == net_sim

    # Smoke test
    net_sim.plot_cells(show=False)

    # Checking Saving unsimulated network
    net.write(tmp_path / 'net_unsim.hdf5', save_unsimulated=True)
    net_unsim_read = read_network(tmp_path / 'net_unsim.hdf5')
    net_unsim = net.copy()
    net_unsim.cell_response = None
    assert net_unsim_read == net_unsim

    # Running simulation on the read unsimulated network and check it against
    # previous simulation
    dpls3 = simulate_dipole(net_unsim_read, tstop=2, n_trials=1, dt=0.5)
    for dpl1, dpl3 in zip(dpls1, dpls3):
        assert_allclose(dpl1.times, dpl3.times, rtol=0.00051, atol=0)
        for dpl_key in dpl1.data.keys():
            assert_allclose(dpl1.data[dpl_key],
                            dpl3.data[dpl_key], rtol=0.000051, atol=0)

    # Smoke test
    net_unsim_read.plot_cells(show=False)

    # Checking reading of raw network
    net_raw = read_network(tmp_path / 'net_sim.hdf5',
                           read_raw=True)
    assert net_raw == net_unsim
    # Checking simulation correctness of read raw network
    dpls4 = simulate_dipole(net_raw, tstop=2, n_trials=1, dt=0.5)
    for dpl1, dpl4 in zip(dpls1, dpls4):
        assert_allclose(dpl1.times, dpl4.times, rtol=0.00051, atol=0)
        for dpl_key in dpl1.data.keys():
            assert_allclose(dpl1.data[dpl_key],
                            dpl4.data[dpl_key], rtol=0.000051, atol=0)

    # Smoke test
    net_raw.plot_cells(show=False)

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

    # Add test to check weights are equal in connections and drives (todo)

    # Tests for checking docstrings
    # check attribute in docstring for write
    docstring_write = net.write.__doc__
    docstring_read = read_network.__doc__
    for attr in net.__dict__.keys():
        if not attr.startswith('_'):
            assert docstring_write.find(attr) != -1
            assert docstring_read.find(attr) != -1
