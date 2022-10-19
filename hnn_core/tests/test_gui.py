# Authors: Huzi Cheng <hzcheng15@icloud.com>
import matplotlib
import numpy as np
import pytest
from hnn_core import Dipole, Network, Params
from hnn_core.gui import HNNGUI
from hnn_core.gui._viz_manager import _idx2figname, _no_overlay_plot_types
from hnn_core.gui.gui import _init_network_from_widgets
from hnn_core.network import pick_connection
from hnn_core.network_models import jones_2009_model
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
    file1_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn-core/master/hnn_core/param/default.json" # noqa
    file2_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn-core/master/hnn_core/param/gamma_L5weak_L2weak.json" # noqa
    gui._simulate_upload_connectivity(file1_url)
    gui._simulate_upload_drives(file1_url)

    # check if parameter is reloaded.
    assert gui.widget_tstop.value == original_tstop
    assert gui.widget_dt.value == original_tstep
    assert len(gui.drive_widgets) == original_drive_count

    # check parameters with different files.
    # file1: connectivity file2: drives
    gui._simulate_upload_connectivity(file1_url)
    assert gui.widget_tstop.value == 170.
    assert gui.connectivity_widgets[0][0].children[1].value == 0.02
    gui._simulate_upload_drives(file2_url)
    assert gui.widget_tstop.value == 250.
    # uploading new drives does not influence the existing connectivity.
    assert gui.connectivity_widgets[0][0].children[1].value == 0.02

    # file2: connectivity file1: drives
    gui._simulate_upload_connectivity(file2_url)
    # now connectivity is refreshed.
    assert gui.connectivity_widgets[0][0].children[1].value == 0.01
    assert gui.drive_widgets[-1]['tstop'].value == 250.
    gui._simulate_upload_drives(file1_url)
    assert gui.connectivity_widgets[0][0].children[1].value == 0.01
    assert gui.drive_widgets[-1]['tstop'].value == 0.


def test_gui_change_connectivity():
    """Test if GUI properly changes cell connectivity parameters."""
    gui = HNNGUI()
    _ = gui.compose()

    for connectivity_slider in gui.connectivity_widgets:
        for vbox in connectivity_slider:
            for w_val in (0.2, 0.9):
                _single_simulation = {}
                _single_simulation['net'] = jones_2009_model(gui.params)
                # specify connection
                conn_indices = pick_connection(
                    net=_single_simulation['net'],
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
                                           _single_simulation,
                                           gui.drive_widgets,
                                           gui.connectivity_widgets,
                                           add_drive=False)

                # test if the new value is reflected in the network
                assert _single_simulation['net'].connectivity[conn_idx][
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
    _ = gui.compose()
    # now the default parameter has been loaded.
    _single_simulation = {}
    _single_simulation['net'] = jones_2009_model(gui.params)
    _init_network_from_widgets(gui.params, gui.widget_dt, gui.widget_tstop,
                               _single_simulation, gui.drive_widgets,
                               gui.connectivity_widgets)

    # copied from test_network.py
    assert np.isclose(_single_simulation['net']._inplane_distance, 1.)
    assert np.isclose(_single_simulation['net']._layer_separation, 1307.4)


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
    default_name = gui.widget_simulation_name.value
    dpls = gui.simulation_data[default_name]['dpls']
    assert isinstance(gui.simulation_data[default_name]["net"], Network)
    assert isinstance(dpls, list)
    assert all([isinstance(dpl, Dipole) for dpl in dpls])


def test_gui_run_simulations():
    """Test if run button triggers multiple simulations correctly."""
    gui = HNNGUI()
    app_layout = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    assert app_layout is not None
    assert gui.widget_backend_selection.value == "Joblib"
    val_tstop = 20
    val_ntrials = 1
    sim_count = 0
    for val_tstop in (10, 12):
        for val_ntrials in (1, 2):
            for val_tstep in (0.05, 0.08):
                gui.widget_simulation_name.value = str(sim_count)
                gui.widget_tstop.value = val_tstop
                gui.widget_ntrials.value = val_ntrials
                gui.widget_dt.value = val_tstep

                gui.run_button.click()
                sim_name = gui.widget_simulation_name.value
                dpls = gui.simulation_data[sim_name]['dpls']

                assert isinstance(gui.simulation_data[sim_name]["net"],
                                  Network)
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

                sim_count += 1

    assert len(list(gui.simulation_data)) == sim_count

    # make sure different simulations must have distinct names
    gui = HNNGUI()
    _ = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    sim_name = gui.widget_simulation_name.value

    gui.run_button.click()
    dpls = gui.simulation_data[sim_name]['dpls']
    assert isinstance(gui.simulation_data[sim_name]["net"], Network)
    assert isinstance(dpls, list)
    assert gui._simulation_status_bar.value == \
        gui._simulation_status_contents['finished']

    gui.widget_simulation_name.value = sim_name
    gui.run_button.click()
    assert len(gui.simulation_data) == 1
    assert gui._simulation_status_bar.value == \
        gui._simulation_status_contents['failed']


def test_gui_take_screenshots():
    """Test if the GUI correctly generates screenshots."""
    gui = HNNGUI()
    gui.compose(return_layout=False)
    screenshot = gui.capture(render=False)
    assert type(screenshot) is IFrame
    gui._simulate_left_tab_click("External drives")
    screenshot1 = gui.capture(render=False)
    assert screenshot._repr_html_() != screenshot1._repr_html_()


def test_gui_add_figure():
    """Test if the GUI adds/deletes figs properly."""
    gui = HNNGUI()
    _ = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    fig_tabs = gui.viz_manager.figs_tabs
    axes_config_tabs = gui.viz_manager.axes_config_tabs
    assert len(fig_tabs.children) == 0
    assert len(axes_config_tabs.children) == 0

    # after each run we should have a default fig
    gui.run_button.click()
    assert len(fig_tabs.children) == 1
    assert len(axes_config_tabs.children) == 1

    assert gui.viz_manager.fig_idx['idx'] == 2

    for idx in range(3):
        n_fig = idx + 2
        gui.viz_manager.make_fig_button.click()
        assert len(fig_tabs.children) == n_fig
        assert len(axes_config_tabs.children) == n_fig

    # we should have 4 figs here
    # delete the 2nd and test if the total number and fig names match or not.
    tmp_fig_idx = 2
    tab_index = tmp_fig_idx - 1
    assert gui.viz_manager.fig_idx['idx'] == 5
    # test delete figures
    axes_config_tabs.children[tab_index].children[0].click()
    assert gui.viz_manager.fig_idx['idx'] == 5

    assert len(fig_tabs.children) == 3
    assert len(axes_config_tabs.children) == 3
    remaining_titles1 = [
        fig_tabs.get_title(idx) for idx in range(len(fig_tabs.children))
    ]
    remaining_titles2 = [
        axes_config_tabs.get_title(idx)
        for idx in range(len(axes_config_tabs.children))
    ]
    correct_remaining_titles = [_idx2figname(idx) for idx in (1, 3, 4)]
    assert remaining_titles1 == remaining_titles2 == correct_remaining_titles


def test_gui_edit_figure():
    """Test if the GUI adds/deletes figs properly."""
    gui = HNNGUI()
    _ = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    fig_tabs = gui.viz_manager.figs_tabs
    axes_config_tabs = gui.viz_manager.axes_config_tabs

    # after each run we should have a default fig
    sim_names = ["t1", "t2", "t3"]
    for sim_idx, sim_name in enumerate(sim_names):
        gui.widget_simulation_name.value = sim_name
        gui.run_button.click()
        print(len(fig_tabs.children), sim_idx)
        n_figs = sim_idx + 1
        assert len(fig_tabs.children) == n_figs
        assert len(axes_config_tabs.children) == n_figs

        axes_config = axes_config_tabs.children[-1].children[1]
        simulation_selection = axes_config.children[0].children[0]
        assert simulation_selection.options == tuple(sim_names[: n_figs])


def test_gui_figure_overlay():
    """Test if the GUI adds/deletes figs properly."""
    gui = HNNGUI()
    _ = gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3

    axes_config_tabs = gui.viz_manager.axes_config_tabs

    gui.run_button.click()
    for tab in axes_config_tabs.children:
        for controls in tab.children[1].children:
            add_plot_button = controls.children[-2].children[0]
            clear_ax_button = controls.children[-2].children[1]
            plot_type_selection = controls.children[1]

            assert plot_type_selection.disabled is True
            clear_ax_button.click()
            # after clearing the axis, we should be able to select plot type.
            assert plot_type_selection.disabled is False

            # disable overlay for certain plot types
            for plot_type in _no_overlay_plot_types:
                plot_type_selection.value = plot_type
                add_plot_button.click()
                assert add_plot_button.disabled is True
                clear_ax_button.click()
                assert add_plot_button.disabled is False
