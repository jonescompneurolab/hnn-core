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
                              serialize_simulation,
                              serialize_config)
from hnn_core.network import pick_connection, _compare_lists
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil
from hnn_core.hnn_io import dict_to_network, read_network_configuration
from IPython.display import IFrame
from ipywidgets import Tab, Text, link

matplotlib.use('agg')
hnn_core_root = Path(__file__).parents[1]
assets_path = Path(hnn_core_root, 'tests', 'assets')


@pytest.fixture
def setup_gui():
    gui = HNNGUI(
        network_configuration=assets_path / 'jones2009_3x3_drives.json'
    )
    gui.compose()
    gui.widget_dt.value = 0.5  # speed up tests
    gui.widget_tstop.value = 70  # speed up tests
    return gui


def check_equal_networks(net1, net2):
    """Checks for equivalency networks

    GUI and API generated networks should be equal except for certain
    attributes that differ. The Network.__eq__ method will not work for
    comparing GUI-derived networks to API-derived networks. This function
    adapts the __eq__
    """
    def check_equality(item1, item2, message=None):
        assert item1 == item2, message

    # Poisson and Bursty drives will have different tstop. This function
    # passes comparing tstop.
    def check_items(dict1, dict2, ignore_keys=[], message=''):
        for d_key, d_value in dict1.items():
            if d_key not in ignore_keys:
                check_equality(d_value, dict2[d_key],
                               f'{message}{d_key} not equal')

    def check_drive(drive1, drive2, keys):
        name = drive1['name']
        for key in keys:
            value1 = drive1[key]
            value2 = drive2[key]
            if key != 'dynamics':
                check_equality(value1, value2,
                               f'>{name}>{key} not equal')
            else:
                check_items(value1, value2, ignore_keys=['tstop'],
                            message=f'>{name}>{key}>')

    # Check connectivity
    assert len(net1.connectivity) == len(net2.connectivity)
    assert _compare_lists(net1.connectivity, net2.connectivity)

    # Check drives
    for drive1, drive2 in zip(net1.external_drives.values(),
                              net2.external_drives.values()):
        check_drive(drive1, drive2, keys=drive1.keys())

    # Check external biases
    for bias_name, bias_dict in net1.external_biases.items():
        for cell_type, bias_params in bias_dict.items():
            check_items(bias_params,
                        net2.external_biases[bias_name][cell_type],
                        ignore_keys=['tstop'],
                        message=f'{bias_name}>{cell_type}>')

    # Check all other attributes
    attrs_to_ignore = ['connectivity', 'external_drives', 'external_biases']
    for attr in vars(net1).keys():
        if attr.startswith('_') or attr in attrs_to_ignore:
            continue

        check_equality(getattr(net1, attr), getattr(net2, attr),
                       f'{attr} not equal')


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
    default_params = gui.params.copy()

    # clear the connectivity widgets
    original_connectivity_count = len(gui.connectivity_widgets)
    assert original_connectivity_count > 0
    gui.connectivity_widgets = []
    assert len(gui.connectivity_widgets) == 0

    # simulate upload default.json
    file1_path = Path(hnn_core_root, 'param', 'jones2009_base.json')
    file2_path = Path(assets_path, 'gamma_L5weak_L2weak_hierarchical.json')
    gui._simulate_upload_connectivity(file1_path)

    # check if parameter is reloaded.
    assert len(gui.connectivity_widgets) == original_connectivity_count

    # check parameters with different files
    assert gui.connectivity_widgets[0][0].children[1].value == 0.02
    # value should change when loading connectivity from file 2
    gui._simulate_upload_connectivity(file2_path)
    assert gui.connectivity_widgets[0][0].children[1].value == 0.01

    # check that the gui param attribute was updated
    assert gui.params != default_params

    # Load drives and make sure connectivity does not change
    gui._simulate_upload_drives(file1_path)
    assert gui.connectivity_widgets[0][0].children[1].value == 0.01

    plt.close('all')


def test_gui_upload_drives():
    """Test if gui handles uploaded drive parameters correctly"""
    gui = HNNGUI()
    _ = gui.compose()

    # clear the drive widgets
    original_drive_count = len(gui.drive_widgets)
    assert original_drive_count > 0
    gui.delete_drive_button.click()
    assert len(gui.drive_widgets) == 0

    # simulate upload default.json
    file1_url = Path(hnn_core_root, 'param', 'jones2009_base.json')
    file2_url = Path(assets_path, 'gamma_L5weak_L2weak_hierarchical.json')
    file3_url = Path(assets_path, 'jones2009_3x3_drives.json')

    # check if parameter reloads
    gui._simulate_upload_drives(file1_url)
    assert len(gui.drive_widgets) == original_drive_count
    drive_types = [widget['type'] for widget in gui.drive_widgets]
    assert drive_types == ['Evoked', 'Evoked', 'Evoked']

    # check parameters with different files.
    gui._simulate_upload_drives(file2_url)
    assert len(gui.drive_widgets) == 1
    assert gui.drive_widgets[0]['type'] == 'Poisson'

    # tstop is currently set to the tstop widget because the Network configs
    # do not currently save a universal tstop attribute. In this case
    # the drive tstop gets set to the widget value if the drive stop is larger
    # than the widget tstop. This may change in the future if tstop is saved to
    # the network configs.
    assert gui.drive_widgets[0]['tstop'].value == 170.

    # Load connectivity and make sure drives did not change
    gui._simulate_upload_connectivity(file1_url)
    assert len(gui.drive_widgets) == 1

    # Load file with more drives and make sure it's in the right order
    gui.delete_drive_button.click()
    gui._simulate_upload_drives(file3_url)
    drive_names = [widget['name'] for widget in gui.drive_widgets]
    assert drive_names == ['evdist1', 'evprox1', 'evprox2',
                           'alpha_prox', 'poisson', 'tonic']

    # Check for correct tonic bias loading
    assert gui.drive_widgets[5]['type'] == 'Tonic'
    assert gui.drive_widgets[5]['amplitude']['L2_pyramidal'].value == 1.0
    assert gui.drive_widgets[5]['amplitude']['L5_basket'].value == 0.0
    assert gui.drive_widgets[5]['tstop'].value == 170.0

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
                _single_simulation['net'] = dict_to_network(gui.params)
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


def test_gui_init_network(setup_gui):
    """Test if gui initializes network properly"""
    gui = setup_gui
    # now the default parameter has been loaded.
    _single_simulation = {}
    _single_simulation['net'] = dict_to_network(gui.params)
    _init_network_from_widgets(gui.params, gui.widget_dt, gui.widget_tstop,
                               _single_simulation, gui.drive_widgets,
                               gui.connectivity_widgets,
                               gui.cell_pameters_widgets)
    plt.close('all')

    net_from_gui = _single_simulation['net']

    # copied from test_network.py
    assert np.isclose(net_from_gui._inplane_distance, 1.)
    assert np.isclose(net_from_gui._layer_separation, 1307.4)

    # Compare Network created from API
    config_path = assets_path / 'jones2009_3x3_drives.json'
    net_from_api = read_network_configuration(config_path)

    check_equal_networks(net_from_gui, net_from_api)


@requires_mpi4py
@requires_psutil
def test_gui_run_simulation_mpi():
    """Test if run button triggers simulation with MPIBackend."""
    gui = HNNGUI()
    _ = gui.compose()

    gui.widget_tstop.value = 70
    gui.widget_dt.value = 0.5
    gui.widget_backend_selection.value = "MPI"
    gui.widget_ntrials.value = 2
    gui.run_button.click()

    default_name = gui.widget_simulation_name.value
    dpls = gui.simulation_data[default_name]['dpls']
    assert isinstance(gui.simulation_data[default_name]["net"], Network)
    assert isinstance(dpls, list)
    assert len(dpls) > 0
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

    # Set cell_specific to False
    gui.drive_widgets[0]['is_cell_specific'].value = False
    # Check that the n_drive_cells is not disabled
    assert not gui.drive_widgets[0]['n_drive_cells'].disabled

    # Get name of first drive
    drive_name = gui.drive_widgets[0]['name']

    # Loop by number of drive cells
    for i, n_drive_cells in enumerate([1, 3]):
        gui.widget_simulation_name.value = f'sim_{i}'
        gui.drive_widgets[0]['n_drive_cells'].value = n_drive_cells

        # Run simulation
        gui.run_button.click()
        sim = (gui.viz_manager.data
               ['simulations'][gui.widget_simulation_name.value])

        # Filter connections for specific driver_name first
        network_connections = sim['net'].connectivity
        driver_connections = [conn for conn in network_connections
                              if conn['src_type'] == drive_name]

        # Check src_gids length
        for connectivity in driver_connections:
            assert len(connectivity['src_gids']) == n_drive_cells


def test_gui_cell_specific_drive(setup_gui):
    """Tests 1:1 connection with cell_specific widget"""
    gui = setup_gui
    # Set cell_specific to False
    gui.drive_widgets[0]['is_cell_specific'].value = True
    # Assert that the n_drive_cells is disabled
    assert gui.drive_widgets[0]['n_drive_cells'].disabled

    # Get name of first drive
    driver_name = gui.drive_widgets[0]['name']

    # Run simulation
    gui.run_button.click()
    sim = gui.viz_manager.data['simulations'][gui.widget_simulation_name.value]

    # Filter connections for specific driver_name first
    network_connections = sim['net'].connectivity
    driver_connections = [conn for conn in network_connections
                          if conn['src_type'] == driver_name]

    # Check src_gids length
    for connectivity in driver_connections:
        assert (len(connectivity['src_gids']) ==
                len(connectivity['target_gids']))


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
    """Tests updating a figure creates plots with data."""

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


def test_dipole_data_overlay(setup_gui):
    """Tests dipole plot with a simulation and data overlay."""
    gui = setup_gui

    # Run simulation with 2 trials
    gui.widget_ntrials.value = 2
    gui.run_button.click()

    # Load data
    file_path = assets_path / 'test_default.csv'
    gui._simulate_upload_data(file_path)

    # Edit the figure with data overlay
    figid = 1
    figname = f'Figure {figid}'
    axname = 'ax1'
    gui._simulate_viz_action("edit_figure", figname,
                             axname, 'default', 'current dipole', {}, 'clear')
    gui._simulate_viz_action("edit_figure", figname,
                             axname, 'default', 'current dipole',
                             {'data_to_compare': 'test_default'},
                             'plot')
    ax = gui.viz_manager.figs[figid].axes[1]

    # Check number of lines
    # 2 trials, 1 average, 2 data (data is over-plotted twice for some reason)
    # But it only appears in the legend once.
    assert len(ax.lines) == 5
    assert len(ax.legend_.texts) == 2
    assert ax.legend_.texts[0]._text == 'default: average'
    assert ax.legend_.texts[1]._text == 'test_default'

    # Check RMSE is printed
    assert 'RMSE(default, test_default):' in ax.texts[0]._text

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


def test_gui_download_configuration(setup_gui):
    """Test the GUI download simulation pipeline."""

    gui = setup_gui

    # Initiate 1st simulation
    sim_name = "sim1"
    gui.widget_simulation_name.value = sim_name

    # Run simulation
    gui.run_button.click()

    # serialize configurations of the simulation
    configs = serialize_config(gui.data, sim_name)
    net_from_buffer = json.loads(configs)

    # Load configuration from file
    source_network_config = assets_path / 'jones2009_3x3_drives.json'
    with open(source_network_config, 'r') as file:
        net_source_config = json.load(file)

    # Create  networks
    net1 = dict_to_network(net_from_buffer)
    net2 = dict_to_network(net_source_config)

    check_equal_networks(net1, net2)


def test_gui_add_tonic_input():
    """Test if gui add different type of drives."""
    gui = HNNGUI()
    _ = gui.compose()
    assert 'tonic' not in [drive['type'].lower()
                           for drive in gui.drive_widgets]

    _single_simulation = {}
    _single_simulation['net'] = dict_to_network(gui.params)

    # Add tonic input widget
    gui.widget_drive_type_selection.value = "Tonic"
    gui.add_drive_button.click()

    # Check last drive (Tonic)
    last_drive = gui.drive_widgets[-1]
    assert last_drive['type'] == "Tonic"
    assert last_drive['t0'].value == 0.0
    assert last_drive['tstop'].value == 170.0
    assert last_drive['amplitude']['L5_pyramidal'].value == 0

    # Set new widget values
    last_drive['t0'].value = 0
    last_drive['tstop'].value = 15
    last_drive['amplitude']['L5_pyramidal'].value = 10

    # Check that you can't add more than one tonic
    gui.add_drive_button.click()
    assert ([drive['type'].lower() for drive in gui.drive_widgets] ==
            ['evoked', 'evoked', 'evoked', 'tonic'])

    # Add tonic bias to the network
    _init_network_from_widgets(gui.params, gui.widget_dt, gui.widget_tstop,
                               _single_simulation, gui.drive_widgets,
                               gui.connectivity_widgets,
                               gui.cell_pameters_widgets)

    net = _single_simulation['net']
    assert net.external_biases['tonic'] is not None
    assert net.external_biases['tonic']['L5_pyramidal']['t0'] == 0.0
    assert net.external_biases['tonic']['L5_pyramidal']['tstop'] == 15.0
    assert net.external_biases['tonic']['L5_pyramidal']['amplitude'] == 10.0


def test_gui_cell_params_widgets(setup_gui):
    """Test if gui add different type of drives."""
    gui = setup_gui
    _single_simulation = {}
    _single_simulation['net'] = dict_to_network(gui.params)
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
    assert (len(cell_params['Geometry L2']) == 20)
    assert (len(cell_params['Geometry L5']) == 22)
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


def test_delete_single_drive(setup_gui):
    """Deleting a single drive."""
    gui = setup_gui
    assert len(gui.drive_accordion.children) == 6
    assert gui.drive_accordion.titles == ('evdist1 (distal)',
                                          'evprox1 (proximal)',
                                          'evprox2 (proximal)',
                                          'alpha_prox (proximal)',
                                          'poisson (proximal)',
                                          'tonic')

    gui._simulate_delete_single_drive(2)
    assert len(gui.drive_accordion.children) == 5
    assert gui.drive_accordion.titles == ('evdist1 (distal)',
                                          'evprox1 (proximal)',
                                          'alpha_prox (proximal)',
                                          'poisson (proximal)',
                                          'tonic')
