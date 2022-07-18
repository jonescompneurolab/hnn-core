# Authors: Huzi Cheng <hzcheng15@icloud.com>
import numpy as np
import os.path as op

import hnn_core
from hnn_core import Dipole, Network, Params
from hnn_core.gui.gui import HNNGUI, _init_network_from_widgets


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
    for sliders in gui.connectivity_sliders:
        for slider in sliders:
            float_text, slider, _ = slider.children
            for val in (0.2, 0.4, 0.9):
                float_text.value = val
                assert slider.value == val
            for val in (0.2, 0.4, 0.9):
                slider.value = val
                assert float_text.value == val


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
                               gui.drive_widgets)

    # copied from test_network.py
    assert np.isclose(gui.variables['net']._inplane_distance, 1.)  # default
    assert np.isclose(gui.variables['net']._layer_separation, 1307.4)


def test_run_gui():
    """Test if run button triggers simulation."""
    gui = HNNGUI()
    app_layout = gui.run()
    assert app_layout is not None
    gui.run_button.click()
    dpls = gui.variables['dpls']
    assert isinstance(gui.variables["net"], Network)
    assert isinstance(dpls, list)
    assert all([isinstance(dpl, Dipole) for dpl in dpls])

# For other tests, we can follow this paradigm:
# 1. Initialize the gui
# 2. Change some parameters from the widget interface.
# 3. Use modified parameters to construct the Network model
# 4. (Optional) run simulation
# 5. Check if the simulation results or network with modified parameters
#    match.

# def test_gui_run_simulation():
#     """Test if gui can reproduce the same results as using hnn-core"""
#     pass

# def test_gui_update_simulation_parameters():
#     """Test if gui builds new network model after changing simulation
#     parameters."""
#     pass

# def test_gui_update_cell_connectivity():
#     """Test if gui builds new network model after changing simulation
#     parameters."""
#     pass

# def test_gui_update_drives():
#     """Test if gui builds new network model after changing simulation
#     parameters."""
#     pass
