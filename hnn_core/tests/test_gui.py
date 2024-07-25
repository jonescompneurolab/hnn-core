# Authors: Huzi Cheng <hzcheng15@icloud.com>
#          Camilo Diaz <camilo_diaz@brown.edu>
#          George Dang <george_dang@brown.edu>
import codecs
import io
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import traitlets
import os

from pathlib import Path
from hnn_core import Dipole, Network
from hnn_core.gui import HNNGUI
from hnn_core.gui._viz_manager import (_idx2figname,
                                       _plot_types,
                                       _no_overlay_plot_types,
                                       unlink_relink)
from hnn_core.gui.gui import (_init_network_from_widgets,
                              _prepare_upload_file,
                              _update_nested_dict,
                              serialize_simulation)
from hnn_core.network import pick_connection
from hnn_core.network_models import jones_2009_model
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil
from IPython.display import IFrame
from ipywidgets import Tab, Text, link

matplotlib.use('agg')
hnn_core_root = Path(__file__).parents[1]
assets_path = Path(hnn_core_root, 'tests', 'assets')


@pytest.fixture
def setup_gui():
    gui = HNNGUI()
    gui.compose()
    gui.params['N_pyr_x'] = 3
    gui.params['N_pyr_y'] = 3
    gui.widget_dt.value = 0.5  # speed up tests
    gui.widget_tstop.value = 70  # speed up tests
    return gui


def test_gui_load_params():
    """Test if gui loads default parameters properly"""
    gui = HNNGUI()

    assert isinstance(gui.params, dict)
    assert gui.params['object_type'] == 'Network'
    plt.close('all')


def test_gui_compose():
    gui = HNNGUI()
    gui.compose()
    assert len(gui.connectivity_widgets) == 12
    assert len(gui.drive_widgets) == 3
    plt.close('all')


def test_prepare_upload_file():
    """Tests that input files from local or url sources import correctly"""
    def _import_json(content):
        decode = codecs.decode(content, encoding="utf-8")
        json_content = json.load(io.StringIO(decode))
        return json_content

    url = "https://raw.githubusercontent.com/jonescompneurolab/hnn-core/master/hnn_core/param/default.json"  # noqa
    file = Path(hnn_core_root, 'param', 'default.json')

    content_from_url = _prepare_upload_file(url)[0]
    content_from_local = _prepare_upload_file(file)[0]

    assert (content_from_url['name'] ==
            content_from_local['name'] ==
            'default.json')
    assert (content_from_url['type'] ==
            content_from_local['type'] ==
            'application/json')
    # Check that the size attribute is present. Cannot do an equivalency check
    # because file systems may add additional when saving to disk.
    assert 'size' in content_from_url
    assert 'size' in content_from_local

    # Check that the content is the same when imported as dict
    dict_from_url = _import_json(content_from_url.get('content'))
    dict_from_local = _import_json(content_from_local.get('content'))
    assert dict_from_url == dict_from_local


def test_gui_upload_connectivity():
    """Test if gui handles uploaded connectivity parameters correctly"""
    gui = HNNGUI()
    _ = gui.compose()
    # clear the drive and connectivity widgets
    original_connectivity_count = len(gui.connectivity_widgets)
    assert original_connectivity_count > 0
    gui.connectivity_widgets = []
    assert len(gui.connectivity_widgets) == 0

    # simulate upload default.json
    file1_path = Path(hnn_core_root, 'param', 'jones2009_base.json')
    file2_path = Path(hnn_core_root, 'param',
                      'gamma_L5weak_L2weak_hierarchical.json')
    gui._simulate_upload_connectivity(file1_path)

    # check if parameter is reloaded.
    assert len(gui.connectivity_widgets) == original_connectivity_count

    # check parameters with different files
    gui._simulate_upload_connectivity(file1_path)
    assert gui.connectivity_widgets[0][0].children[1].value == 0.02
    # value should change when loading connectivity from file 2
    gui._simulate_upload_connectivity(file2_path)
    assert gui.connectivity_widgets[0][0].children[1].value == 0.01

    # TODO Add check that loading drives does not change connectivity

    plt.close('all')

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
    file1_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn-core/master/hnn_core/param/default.json"  # noqa
    file2_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn-core/master/hnn_core/param/gamma_L5weak_L2weak.json"  # noqa
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
    plt.close('all')


def test_gui_upload_data():
    """Test if gui handles uploaded data"""
    gui = HNNGUI()
    _ = gui.compose()

    assert len(gui.viz_manager.data['figs']) == 0
    assert len(gui.data['simulation_data']) == 0

    file1_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/MEG_detection_data/S1_SupraT.txt"  # noqa
    file2_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/MEG_detection_data/yes_trial_S1_ERP_all_avg.txt"  # noqa
    gui._simulate_upload_data(file1_url)
    assert len(gui.data['simulation_data']) == 1
    assert 'S1_SupraT' in gui.data['simulation_data'].keys()
    assert gui.data['simulation_data']['S1_SupraT']['net'] is None
    assert type(gui.data['simulation_data']['S1_SupraT']['dpls']) is list
    assert len(gui.viz_manager.data['figs']) == 1
    # support uploading multiple external data.
    gui._simulate_upload_data(file2_url)
    assert len(gui.data['simulation_data']) == 2
    assert len(gui.viz_manager.data['figs']) == 2

    # make sure no repeated uploading for the same name.
    gui._simulate_upload_data(file1_url)
    assert len(gui.data['simulation_data']) == 2
    assert len(gui.viz_manager.data['figs']) == 2

    # No data loading for legacy multi-trial data files.
    file3_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/gamma_tutorial/100_trials.txt"  # noqa
    with pytest.raises(
            ValueError,
            match="Data are supposed to have 2 or 4 columns while we have 101."
    ):
        gui._simulate_upload_data(file3_url)
    assert len(gui.data['simulation_data']) == 2
    assert len(gui.viz_manager.data['figs']) == 2

    plt.close('all')


def test_gui_change_connectivity():
    """Test if GUI properly changes cell connectivity parameters."""
    gui = HNNGUI()
    _ = gui.compose()

    for connectivity_field in gui.connectivity_widgets:
        for vbox in connectivity_field:
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

                # re initialize network
                _init_network_from_widgets(gui.params, gui.widget_dt,
                                           gui.widget_tstop,
                                           _single_simulation,
                                           gui.drive_widgets,
                                           gui.connectivity_widgets,
                                           gui.cell_pameters_widgets,
                                           add_drive=False)

                # test if the new value is reflected in the network
                assert (_single_simulation['net'].connectivity[conn_idx]
                        ['nc_dict']['A_weight'] == w_val)
    plt.close('all')


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
    plt.close('all')


def test_gui_init_network():
    """Test if gui initializes network properly"""
    gui = HNNGUI()
    _ = gui.compose()
    # now the default parameter has been loaded.
    _single_simulation = {}
    _single_simulation['net'] = jones_2009_model(gui.params)
    _init_network_from_widgets(gui.params, gui.widget_dt, gui.widget_tstop,
                               _single_simulation, gui.drive_widgets,
                               gui.connectivity_widgets,
                               gui.cell_pameters_widgets)
    plt.close('all')

    # copied from test_network.py
    assert np.isclose(_single_simulation['net']._inplane_distance, 1.)
    assert np.isclose(_single_simulation['net']._layer_separation, 1307.4)


@requires_mpi4py
@requires_psutil
def test_gui_run_simulation_mpi(setup_gui):
    """Test if run button triggers simulation with MPIBackend."""
    gui = setup_gui

    gui.widget_backend_selection.value = "MPI"
    gui.run_button.click()
    default_name = gui.widget_simulation_name.value
    dpls = gui.simulation_data[default_name]['dpls']
    assert isinstance(gui.simulation_data[default_name]["net"], Network)
    assert isinstance(dpls, list)
    assert all([isinstance(dpl, Dipole) for dpl in dpls])
    plt.close('all')


def test_gui_run_simulations(setup_gui):
    """Test if run button triggers multiple simulations correctly."""
    gui = setup_gui

    tstop_trials_tstep = [(10, 1, 0.25),
                          (10, 2, 0.5),
                          (12, 1, 0.5)]
    assert gui.widget_backend_selection.value == "Joblib"
    sim_count = 0

    for val_tstop, val_ntrials, val_tstep in tstop_trials_tstep:
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


def test_non_unique_name_error(setup_gui):
    """ Checks that simulation fails if new name is not supplied. """
    gui = setup_gui

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
    plt.close('all')


def test_gui_take_screenshots():
    """Test if the GUI correctly generates screenshots."""
    gui = HNNGUI()
    gui.compose(return_layout=False)
    screenshot = gui.capture(render=False)
    assert type(screenshot) is IFrame
    gui._simulate_left_tab_click("External drives")
    screenshot1 = gui.capture(render=False)
    assert screenshot._repr_html_() != screenshot1._repr_html_()
    plt.close('all')


def test_gui_add_figure(setup_gui):
    """Test if the GUI adds/deletes figs properly."""
    gui = setup_gui

    fig_tabs = gui.viz_manager.figs_tabs
    axes_config_tabs = gui.viz_manager.axes_config_tabs
    assert len(fig_tabs.children) == 0
    assert len(axes_config_tabs.children) == 0

    # after each run we should have a default fig
    gui.run_button.click()
    assert len(fig_tabs.children) == 1
    assert len(axes_config_tabs.children) == 1
    assert gui.viz_manager.fig_idx['idx'] == 2

    # Check default figs have data on their axis
    assert gui.viz_manager.figs[1].axes[0].has_data()
    assert gui.viz_manager.figs[1].axes[1].has_data()

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
    plt.close('all')


def test_gui_add_data_dependent_figure(setup_gui):
    """Test if the GUI adds/deletes figs data dependent properly."""
    gui = setup_gui

    fig_tabs = gui.viz_manager.figs_tabs
    axes_config_tabs = gui.viz_manager.axes_config_tabs
    assert len(fig_tabs.children) == 0
    assert len(axes_config_tabs.children) == 0

    # after each run we should have a default fig
    gui.run_button.click()
    assert len(fig_tabs.children) == 1
    assert len(axes_config_tabs.children) == 1
    assert gui.viz_manager.fig_idx['idx'] == 2

    template_names = [('Drive-Dipole (2x1)', 2),
                      ('Dipole Layers (3x1)', 3),
                      ('Drive-Spikes (2x1)', 2),
                      ('Dipole-Spectrogram (2x1)', 2),
                      ("Dipole-Spikes (2x1)", 2),
                      ('Drive-Dipole-Spectrogram (3x1)', 3),
                      ('PSD Layers (3x1)', 3)]

    n_fig = 1
    for template_name, num_axes in template_names:
        gui.viz_manager.templates_dropdown.value = template_name
        assert len(gui.viz_manager.datasets_dropdown.options) == 1
        gui.viz_manager.make_fig_button.click()
        # Check  figs have data on their axis
        for ax in range(num_axes):
            assert gui.viz_manager.figs[n_fig + 1].axes[ax].has_data()
        n_fig = n_fig + 1

    # test number of created figures
    assert len(fig_tabs.children) == n_fig


def test_gui_edit_figure(setup_gui):
    """Test if the GUI adds/deletes figs properly."""
    gui = setup_gui

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
        simulation_selection = axes_config.children[0].children[1]
        assert simulation_selection.options == tuple(sim_names[:n_figs])
    plt.close('all')


def test_gui_synchronous_inputs(setup_gui):
    """Test if the GUI creates plot using synchronous_inputs."""
    gui = setup_gui

    # set synch inputs to first driver in simulation
    driver_name = gui.drive_widgets[0]['name']
    gui.drive_widgets[0]['is_synch_inputs'].value = True

    # Run simulation
    gui.run_button.click()

    sim = gui.viz_manager.data['simulations']['default']
    network_connections = sim['net'].connectivity
    # Filter connections for specific driver_name first
    driver_connections = [conn for conn in network_connections
                          if conn['src_type'] == driver_name]

    # Check src_gids length
    for connectivity in driver_connections:
        assert len(connectivity['src_gids']) == 1


def test_gui_figure_overlay(setup_gui):
    """Test if the GUI adds/deletes figs properly."""
    gui = setup_gui

    axes_config_tabs = gui.viz_manager.axes_config_tabs

    gui.run_button.click()
    for tab in axes_config_tabs.children:
        for controls in tab.children[1].children:
            add_plot_button = controls.children[-2].children[0]
            clear_ax_button = controls.children[-2].children[1]
            plot_type_selection = controls.children[0]

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
    plt.close('all')


def test_gui_adaptive_spectrogram(setup_gui):
    """Test the adaptive spectrogram functionality of the HNNGUI."""
    gui = setup_gui

    gui.run_button.click()
    figid = 1
    figname = f'Figure {figid}'
    axname = 'ax1'
    gui._simulate_viz_action("edit_figure", figname, axname, 'default',
                             'spectrogram', {}, 'clear')
    gui._simulate_viz_action("edit_figure", figname, axname, 'default',
                             'spectrogram', {}, 'plot')
    # make sure the colorbar is correctly added
    assert any(['_cbar-ax-' in attr
                for attr in dir(gui.viz_manager.figs[figid])]) is True
    assert len(gui.viz_manager.figs[1].axes) == 3
    # make sure the colorbar is safely removed
    gui._simulate_viz_action("edit_figure", figname, axname, 'default',
                             'spectrogram', {}, 'clear')
    assert any(['_cbar-ax-' in attr
                for attr in dir(gui.viz_manager.figs[figid])]) is False
    assert len(gui.viz_manager.figs[1].axes) == 2
    plt.close('all')


def test_gui_visualization(setup_gui):
    """ Tests updating a figure creates plots with data. """

    gui = setup_gui
    gui.run_button.click()

    figid = 1
    figname = f'Figure {figid}'
    axname = 'ax1'
    # Spectrogram has a separate test and does not need to be tested here
    gui_plots_no_spectrogram = [s for s in _plot_types if s != 'spectrogram']

    plot_types = ['current dipole',
                  'layer2 dipole',
                  'layer5 dipole',
                  'input histogram',
                  'spikes',
                  'PSD',
                  'network']
    # Make sure all plot types are tested.
    assert len(plot_types) == len(gui_plots_no_spectrogram)
    assert all([name in gui_plots_no_spectrogram for name in plot_types])

    for viz_type in plot_types:
        gui._simulate_viz_action("edit_figure", figname,
                                 axname, 'default', viz_type, {}, 'clear')
        gui._simulate_viz_action("edit_figure", figname,
                                 axname, 'default', viz_type, {}, 'plot')
        # Check if data is plotted on the axes
        assert len(gui.viz_manager.figs[figid].axes) == 2
        # Check default figs have data on their axis
        assert gui.viz_manager.figs[figid].axes[1].has_data()
    plt.close('all')


def test_unlink_relink_widget():
    """Tests the unlinking and relinking of widgets decorator."""

    # Create a basic version of the VizManager class
    class MiniViz:
        def __init__(self):
            self.tab_group_1 = Tab()
            self.tab_group_2 = Tab()
            self.tab_link = link(
                (self.tab_group_1, 'selected_index'),
                (self.tab_group_2, 'selected_index'),
            )

        def add_child(self, to_add=1):
            n_tabs = len(self.tab_group_2.children) + to_add
            # Add tab and select latest tab
            self.tab_group_1.children = \
                [Text(f'Test{s}') for s in np.arange(n_tabs)]
            self.tab_group_1.selected_index = n_tabs - 1

            self.tab_group_2.children = \
                [Text(f'Test{s}') for s in np.arange(n_tabs)]
            self.tab_group_2.selected_index = n_tabs - 1

        @unlink_relink(attribute='tab_link')
        def add_child_decorated(self, to_add):
            self.add_child(to_add)

    # Check that widgets are linked.
    # Error from tab groups momentarily having a different number of children
    gui = MiniViz()
    with pytest.raises(traitlets.TraitError, match='.*index out of bounds.*'):
        gui.add_child(2)

    # Check decorator unlinks and is able to make a change
    gui = MiniViz()
    gui.add_child_decorated(2)
    assert len(gui.tab_group_1.children) == 2
    assert gui.tab_group_1.selected_index == 1
    assert len(gui.tab_group_2.children) == 2
    assert gui.tab_group_2.selected_index == 1

    # Check if the widgets are relinked, the selected index should be synced
    gui.tab_group_1.selected_index = 0
    assert gui.tab_group_2.selected_index == 0


def test_gui_download_simulation(setup_gui):
    """Test the GUI download simulation pipeline."""

    gui = setup_gui

    # Run a simulation with 2 trials
    gui.widget_ntrials.value = 2

    # Initiate 1rs simulation
    sim_name = "sim1"
    gui.widget_simulation_name.value = sim_name

    # Run simulation
    gui.run_button.click()

    _, file_extension = (
        serialize_simulation(gui.data, sim_name))
    # result is a zip file
    assert file_extension == ".zip"

    # Run a simulation with 1 trials
    gui.widget_ntrials.value = 1

    # Initiate 2nd simulation
    sim_name2 = "sim2"
    gui.widget_simulation_name.value = sim_name2

    # Run simulation
    gui.run_button.click()
    _, file_extension = (
        serialize_simulation(gui.data, sim_name2))
    # result is a single csv file
    assert file_extension == ".csv"

    # Check no loaded data is listed in the sims dropdown list to download
    file1_url = "https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/MEG_detection_data/S1_SupraT.txt"  # noqa
    gui._simulate_upload_data(file1_url)
    download_simulation_list = gui.simulation_list_widget.options
    assert (len([sim_name for sim_name in download_simulation_list
                 if sim_name == "S1_SupraT"]) == 0)


def test_gui_upload_csv_simulation(setup_gui):
    """Test if gui handles uploaded csv data"""

    gui = setup_gui

    assert len(gui.viz_manager.data['figs']) == 0
    assert len(gui.data['simulation_data']) == 0

    # Formulate path to the file
    file_path = assets_path / 'test_default.csv'
    absolute_path = str(file_path.resolve())
    if os.name == 'nt':  # Windows
        # Convert backslashes to forward slashes and
        # ensure we have three slashes after 'file:'
        file_url = 'file:///' + absolute_path.replace('\\', '/')
    else:  # UNIX-like systems
        file_url = 'file://' + absolute_path

    _ = gui._simulate_upload_data(file_url)

    # we are loading only 1 trial,
    # assume all the data we need is in the [0] position
    data_lengh = (
        len(gui.data['simulation_data']['test_default']['dpls'][0].times))

    assert len(gui.data['simulation_data']) == 1
    assert 'test_default' in gui.data['simulation_data'].keys()
    assert gui.data['simulation_data']['test_default']['net'] is None
    assert type(gui.data['simulation_data']['test_default']['dpls']) is list
    assert len(gui.viz_manager.data['figs']) == 1
    assert (len(gui.data['simulation_data']['test_default']
                ['dpls'][0].data['agg']) == data_lengh)
    assert (len(gui.data['simulation_data']['test_default']
                ['dpls'][0].data['L2']) == data_lengh)
    assert (len(gui.data['simulation_data']['test_default']
                ['dpls'][0].data['L5']) == data_lengh)


def test_gui_add_tonic_input(setup_gui):
    """Test if gui add different type of drives."""
    gui = setup_gui
    _single_simulation = {}
    _single_simulation['net'] = jones_2009_model(gui.params)

    # Add tonic input widget
    gui.widget_drive_type_selection.value = "Tonic"
    gui.add_drive_button.click()

    # Check last drive (Tonic)
    last_drive_pos = len(gui.drive_widgets) - 1
    assert gui.drive_widgets[last_drive_pos]['type'] == "Tonic"

    gui.drive_widgets[last_drive_pos]['amplitude']["L5_pyramidal"].value = 10
    gui.drive_widgets[last_drive_pos]['t0'].value = 0
    gui.drive_widgets[last_drive_pos]['tstop'].value = 15

    # Add tonic bias to the simulation
    _init_network_from_widgets(gui.params, gui.widget_dt, gui.widget_tstop,
                               _single_simulation, gui.drive_widgets,
                               gui.connectivity_widgets,
                               gui.cell_pameters_widgets)

    assert _single_simulation['net'].external_biases['tonic'] is not None


def test_gui_cell_params_widgets(setup_gui):
    """Test if gui add different type of drives."""
    gui = setup_gui
    _single_simulation = {}
    _single_simulation['net'] = jones_2009_model(gui.params)
    _single_simulation['net'].cell_types
    pyramid_cell_types = [cell_type for cell_type
                          in _single_simulation['net'].cell_types
                          if "pyramidal" in cell_type]
    assert (len(pyramid_cell_types) == 2)

    # Security check for if parameters have been added or removed from the cell
    # params dict. Any additions will need mappings added to the
    # update_{*}_cell_params functions

    layers = gui.cell_layer_radio_buttons.options
    assert (len(layers) == 3)

    keys = gui.cell_pameters_widgets.keys()
    num_cell_params = 0
    for pyramid_cell_type in pyramid_cell_types:
        cell_type = pyramid_cell_type.split('_')[0]
        for cell_layer in layers:
            key = f'{cell_type} Pyramidal_{cell_layer}'
            assert (any(key in k for k in keys))
            num_cell_params += 1

    assert (len(keys) == num_cell_params)

    # Check the if the cell params dictionary has been updated
    cell_params = gui.get_cell_parameters_dict()
    assert (len(cell_params['Geometry']) == 20)
    assert (len(cell_params['Synapses']) == 12)
    assert (len(cell_params['Biophysics L2']) == 10)
    assert (len(cell_params['Biophysics L5']) == 20)


def test_fig_tabs_dropdown_lists(setup_gui):
    """Test the GUI download simulation pipeline."""

    gui = setup_gui

    gui.widget_ntrials.value = 1

    # Initiate 1st simulation
    sim_name = "sim1"
    gui.widget_simulation_name.value = sim_name

    # Run simulation
    gui.run_button.click()

    # Initiate 2nd simulation
    sim_name2 = "sim2"
    gui.widget_simulation_name.value = sim_name2

    # Run simulation
    gui.run_button.click()

    viz_tabs = gui.viz_manager.axes_config_tabs.children
    for tab in viz_tabs:
        controls = tab.children[1]
        for ax_control in controls.children:
            assert ax_control.children[1].description == "Simulation Data:"
            sim_names = ax_control.children[1].options
            # Check that dropdown has been updated with all simulation names
            assert all(sim in sim_names for sim in [sim_name, sim_name2])

            assert ax_control.children[4].description == "Data to Compare:"

            # Check the data to compare dropdown is enable for
            # non "input histograms" plot type
            if ax_control.children[0].value != "input histogram":
                assert not ax_control.children[4].disabled


def test_update_nested_dict():
    """Tests nested dictionary updates values appropriately."""
    original = {'a': 1,
                'b': {'a2': 0,
                      'b2': {'a3': 0
                             }
                      },
                }

    # Changes at each level
    changes = {'a': 2,
               'b': {'a2': 1,
                     'b2': {'a3': 1
                            }
                     },
               }
    updated = _update_nested_dict(original, changes)
    expected = changes
    assert updated == expected

    # Omitted items should not be changed from in the original
    omission = {'a': 2,
                'b': {'a2': 0},
                }
    expected = {'a': 2,
                'b': {'a2': 0,
                      'b2': {'a3': 0
                             }
                      },
                }
    updated = _update_nested_dict(original, omission)
    assert updated == expected

    # Additional items should be added
    addition = {'a': 2,
                'b': {'a2': 0,
                      'b2': {'a3': 0,
                             'b3': 0,
                             },
                      'c2': 1
                      },
                'c': 1
                }
    expected = addition
    updated = _update_nested_dict(original, addition)
    assert updated == expected

    # Test passing of None values
    has_none = {'a': 1,
                'b': {'a2': None},
                }
    # Default behavior will not pass in None values to the update
    expected = original  # No change expected
    updated = _update_nested_dict(original, has_none)
    assert updated == expected
    # Skip_none set of False will pass in None values to the update
    updated = _update_nested_dict(original, has_none, skip_none=False)
    expected = {'a': 1,
                'b': {'a2': None,
                      'b2': {'a3': 0
                             }
                      },
                }
    assert updated == expected

    # Values that evaluate to False that but are not None type should be passed
    # to the updated dict by default.
    has_nulls = {'a': 0,
                 'b': {'a2': np.nan,
                       'b2': {'a3': False,
                              'b3': ''
                              }
                       },
                 }
    # Skip_none set of False will pass in None values to the update
    updated = _update_nested_dict(original, has_nulls)
    expected = has_nulls
    assert updated == expected
