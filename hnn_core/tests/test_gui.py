# Authors: Huzi Cheng <hzcheng15@icloud.com>
import matplotlib
import numpy as np
import pytest
from hnn_core import Dipole, Network, Params
from hnn_core.gui import HNNGUI
from hnn_core.gui.gui import _init_network_from_widgets
from hnn_core.network import pick_connection
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil
from IPython.display import IFrame

matplotlib.use('agg')


def test_gui_load_params():
    """Test if gui loads default parameters properly"""
    gui = HNNGUI()

    assert isinstance(gui.params, Params)

    print(gui.params)
    print(gui.params['L2Pyr*'])


def test_gui_upload_params():
    """Test if gui handles uploaded parameters correctly"""
    gui = HNNGUI()
    _ = gui.compose()

    # change the default loaded parameters
    original_drive_count = len(gui.drive_widgets)
    assert original_drive_count > 0
    gui.delete_drive_button.click()
    assert len(gui.drive_widgets) == 0

    original_tstop = gui.widget_tstop.value
    gui.widget_tstop.value = 1

    original_tstep = gui.widget_dt.value
    gui.widget_dt.value = 1
    # simulate upload default.json
    file_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn-core/master/hnn_core/param/default.json" # noqa
    gui._simulate_upload_file(file_url)

    # check if parameter is reloaded.
    assert gui.widget_tstop.value == original_tstop
    assert gui.widget_dt.value == original_tstep
    assert len(gui.drive_widgets) == original_drive_count


def test_gui_change_connectivity():
    """Test if GUI properly changes cell connectivity parameters."""
    gui = HNNGUI()
    _ = gui.compose()

    for connectivity_slider in gui.connectivity_widgets:
        for vbox in connectivity_slider:
            for w_val in (0.2, 0.9):
                for p_val in (0.1, 1.0):

                    # specify connection
                    conn_indices = pick_connection(
                        net=gui.simulation_data['net'],
                        src_gids=vbox._belongsto['src_gids'],
                        target_gids=vbox._belongsto['target_gids'],
                        loc=vbox._belongsto['location'],
                        receptor=vbox._belongsto['receptor'])

                    assert len(conn_indices) > 0
                    conn_idx = conn_indices[0]

                    # test if the slider and the input field are synchronous
                    vbox.children[1].value = w_val
                    assert vbox.children[2].value == w_val

                    # re initialize network
                    _init_network_from_widgets(gui.params, gui.widget_dt,
                                               gui.widget_tstop,
                                               gui.simulation_data,
                                               gui.drive_widgets,
                                               gui.connectivity_widgets,
                                               add_drive=False)

                    # test if the new value is reflected in the network
                    assert gui.simulation_data['net'].connectivity[conn_idx][
                        'nc_dict']['A_weight'] == w_val


def test_gui_add_drives():
    """Test if gui add different type of drives."""
    gui = HNNGUI()
    _ = gui.compose()

    for val_drive_type in ("Poisson", "Evoked", "Rhythmic"):
        for val_location in ("distal", "proximal"):
            gui.delete_drive_button.click()
            assert len(gui.drive_widgets) == 0

            gui.widget_drive_type_selection.value = val_drive_type
            gui.widget_location_selection.value = val_location
            gui.add_drive_button.click()

            assert len(gui.drive_widgets) == 1
            assert gui.drive_widgets[0]['type'] == val_drive_type
            assert gui.drive_widgets[0]['location'] == val_location
            assert val_drive_type in gui.drive_widgets[0]['name']


def test_gui_init_network():
    """Test if gui initializes network properly"""
    gui = HNNGUI()
    # now the default parameter has been loaded.
    _init_network_from_widgets(gui.params, gui.widget_dt, gui.widget_tstop,
                               gui.simulation_data, gui.drive_widgets,
                               gui.connectivity_widgets)

    # copied from test_network.py
    assert np.isclose(gui.simulation_data['net']._inplane_distance, 1.)
    assert np.isclose(gui.simulation_data['net']._layer_separation, 1307.4)


@requires_mpi4py
@requires_psutil
def test_gui_run_simulation_mpi():
    """Test if run button triggers simulation with MPIBackend."""
    gui = HNNGUI()
    _ = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    gui.widget_backend_selection.value = "MPI"
    gui.widget_tstop.value = 30  # speed up tests
    gui.run_button.click()
    dpls = gui.simulation_data['dpls']
    assert isinstance(gui.simulation_data["net"], Network)
    assert isinstance(dpls, list)
    assert all([isinstance(dpl, Dipole) for dpl in dpls])


def test_gui_run_simulation():
    """Test if run button triggers simulation."""
    gui = HNNGUI()
    app_layout = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    assert app_layout is not None
    assert gui.widget_backend_selection.value == "Joblib"
    val_tstop = 20
    val_ntrials = 1
    for val_tstop in (10, 12):
        for val_ntrials in (1, 2):
            for val_tstep in (0.05, 0.08):
                gui.widget_tstop.value = val_tstop
                gui.widget_ntrials.value = val_ntrials
                gui.widget_dt.value = val_tstep

                gui.run_button.click()

                dpls = gui.simulation_data['dpls']

                assert isinstance(gui.simulation_data["net"], Network)
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


def test_gui_take_screenshots():
    """Test if the GUI correctly generate screenshots."""
    gui = HNNGUI()
    gui.compose(return_layout=False)
    screenshot = gui.capture(render=False)
    assert type(screenshot) is IFrame
    gui.app_layout.left_sidebar.selected_index = 2
    screenshot1 = gui.capture(render=False)
    assert screenshot._repr_html_() != screenshot1._repr_html_()
