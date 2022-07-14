# Authors: Huzi Cheng <hzcheng15@icloud.com>
import numpy as np
from hnn_core.gui.gui import HNNGUI, init_network_from_widgets


def test_run_gui():
    """Test if main gui function gives proper ipywidget"""
    gui = HNNGUI()
    app_layout = gui.run()
    assert app_layout is not None


def test_gui_load_params():
    """Test if gui loads default parameters properly"""
    gui = HNNGUI()
    assert gui.params is not None

    print(gui.params)
    print(gui.params['L2Pyr*'])


def test_gui_init_network():
    """Test if gui initializes network properly"""
    gui = HNNGUI()
    # now the default parameter has been loaded.
    init_network_from_widgets(gui.params, gui.tstep, gui.tstop, gui.variables,
                              gui.drive_widgets)

    # copied from test_network.py
    assert np.isclose(gui.variables['net']._inplane_distance, 1.)  # default
    assert np.isclose(gui.variables['net']._layer_separation, 1307.4)  # default


def test_gui_run_simulation():
    """Test if gui can reproduce the same results as using hnn-core"""
    pass


def test_gui_update_simulation_parameters():
    """Test if gui builds new network model after changing simulation
    parameters."""
    pass


def test_gui_update_cell_connectivity():
    """Test if gui builds new network model after changing simulation
    parameters."""
    pass


def test_gui_update_drives():
    """Test if gui builds new network model after changing simulation
    parameters."""
    pass
