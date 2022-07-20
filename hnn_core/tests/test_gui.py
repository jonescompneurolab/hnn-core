# Authors: Huzi Cheng <hzcheng15@icloud.com>
import os.path as op

import hnn_core
import numpy as np
import pytest
from hnn_core import Dipole, Network, Params
from hnn_core.gui.gui import HNNGUI, _init_network_from_widgets
from hnn_core.network import pick_connection
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil


def test_gui_load_params():
    """Test if gui loads default parameters properly"""
    gui = HNNGUI()

    assert isinstance(gui.params, Params)

    print(gui.params)
    print(gui.params['L2Pyr*'])


def test_gui_upload_params():
    """Test if gui handles uploaded parameters correctly"""
    gui = HNNGUI()
    _ = gui.run()

    params_name = 'default.json'
    hnn_core_root = op.join(op.dirname(hnn_core.__file__))
    params_fname = op.join(hnn_core_root, 'param', params_name)

    content = b""
    with open(params_fname, "rb") as f:
        for line in f:
            pass
            content += line
    uploaded_value = {
        params_name: {
            'metadata': {
                'name': params_name,
                'type': 'application/json',
                'size': len(content),
            },
            'content': content
        }
    }

    # change the default loaded parameters
    original_drive_count = len(gui.drive_widgets)
    assert original_drive_count > 0
    gui.delete_drive_button.click()
    assert len(gui.drive_widgets) == 0

    original_tstop = gui.tstop.value
    gui.tstop.value = 1

    original_tstep = gui.tstep.value
    gui.tstep.value = 1

    # manually send uploaded content
    gui.load_button.set_trait('value', uploaded_value)

    # check if parameter is reloaded.
    assert gui.tstop.value == original_tstop
    assert gui.tstep.value == original_tstep
    assert len(gui.drive_widgets) == original_drive_count


def test_gui_change_connectivity():
    """Test if GUI properly changes cell connectivity parameters.
    """
    gui = HNNGUI()
    _ = gui.run()

    for connectivity_slider in gui.connectivity_sliders:
        for vbox in connectivity_slider:
            for w_val in (0.2, 0.9):
                for p_val in (0.1, 1.0):

                    # specify connection
                    conn_indices = pick_connection(
                        net=gui.variables['net'],
                        src_gids=vbox._belongsto['src_gids'],
                        target_gids=vbox._belongsto['target_gids'],
                        loc=vbox._belongsto['location'],
                        receptor=vbox._belongsto['receptor'])

                    assert len(conn_indices) > 0
                    conn_idx = conn_indices[0]

                    # test if the slider and the input field are synchronous
                    vbox.children[1].value = w_val
                    assert vbox.children[2].value == w_val

                    vbox.children[3].value = p_val

                    # re initialize network
                    _init_network_from_widgets(gui.params, gui.tstep,
                                               gui.tstop, gui.variables,
                                               gui.drive_widgets,
                                               gui.connectivity_sliders,
                                               add_drive=False)

                    # test if the new value is reflected in the network
                    assert gui.variables['net'].connectivity[conn_idx][
                        'nc_dict']['A_weight'] == w_val
                    assert gui.variables['net'].connectivity[conn_idx][
                        'probability'] == p_val


def test_gui_add_drives():
    """Test if gui add different type of drives."""
    gui = HNNGUI()
    _ = gui.run()

    for val_drive_type in ("Poisson", "Evoked", "Rhythmic"):
        for val_location in ("distal", "proximal"):
            gui.delete_drive_button.click()
            assert len(gui.drive_widgets) == 0

            gui.drive_type_selection.value = val_drive_type
            gui.location_selection.value = val_location
            gui.add_drive_button.click()

            assert len(gui.drive_widgets) == 1
            assert gui.drive_widgets[0]['type'] == val_drive_type
            assert gui.drive_widgets[0]['location'] == val_location
            assert val_drive_type in gui.drive_widgets[0]['name']


def test_gui_init_network():
    """Test if gui initializes network properly"""
    gui = HNNGUI()
    # now the default parameter has been loaded.
    _init_network_from_widgets(gui.params, gui.tstep, gui.tstop, gui.variables,
                               gui.drive_widgets, gui.connectivity_sliders)

    # copied from test_network.py
    assert np.isclose(gui.variables['net']._inplane_distance, 1.)  # default
    assert np.isclose(gui.variables['net']._layer_separation, 1307.4)


@requires_mpi4py
@requires_psutil
def test_gui_run_simulation_mpi():
    """Test if run button triggers simulation with MPIBackend."""
    gui = HNNGUI()
    _ = gui.run()
    gui.backend_selection.value = "MPI"
    gui.tstop.value = 30  # speed up tests
    gui.run_button.click()
    dpls = gui.variables['dpls']
    assert isinstance(gui.variables["net"], Network)
    assert isinstance(dpls, list)
    assert all([isinstance(dpl, Dipole) for dpl in dpls])


def test_gui_run_simulation():
    """Test if run button triggers simulation."""
    gui = HNNGUI()
    app_layout = gui.run()
    assert app_layout is not None
    assert gui.backend_selection.value == "Joblib"
    val_tstop = 20
    val_ntrials = 1
    for val_tstop in (10, 12):
        for val_ntrials in (1, 2):
            for val_tstep in (0.05, 0.08):
                gui.tstop.value = val_tstop
                gui.ntrials.value = val_ntrials
                gui.tstep.value = val_tstep

                gui.run_button.click()

                dpls = gui.variables['dpls']

                assert isinstance(gui.variables["net"], Network)
                assert isinstance(dpls, list)
                assert all([isinstance(dpl, Dipole) for dpl in dpls])
                assert len(dpls) == val_ntrials
                assert all([
                    pytest.approx(dpl.times[-1]) == val_tstop for dpl in dpls
                ])
                assert all([
                    pytest.approx(dpl.times[1] - dpl.times[0]) == val_tstep
                    for dpl in dpls
                ])
