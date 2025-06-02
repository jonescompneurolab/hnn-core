import os.path as op

import matplotlib
from matplotlib import backend_bases
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar

import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, jones_2009_model
from hnn_core.viz import (
    plot_cells,
    plot_dipole,
    plot_psd,
    plot_tfr_morlet,
    plot_connectivity_matrix,
    plot_cell_connectivity,
    plot_drive_strength,
    NetworkPlotter,
)
from hnn_core.dipole import simulate_dipole

matplotlib.use("agg")


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    # Code runs after the test finishes
    yield
    plt.close("all")


@pytest.fixture
def setup_net():
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)
    net = jones_2009_model(params, mesh_shape=(3, 3))

    return net


def _fake_click(fig, ax, point, button=1):
    """Fake a click at a point within axes."""
    x, y = ax.transData.transform_point(point)
    button_press_event = backend_bases.MouseEvent(
        name="button_press_event", canvas=fig.canvas, x=x, y=y, button=button
    )
    fig.canvas.callbacks.process("button_press_event", button_press_event)


def test_network_visualization(setup_net):
    """Test network visualisations."""
    net = setup_net
    plot_cells(net)
    ax = net.cell_types["L2_pyramidal"].plot_morphology()
    assert len(ax.lines) == 8

    conn_idx = 0
    plot_connectivity_matrix(net, conn_idx, show=False)
    with pytest.raises(TypeError, match="net must be an instance of"):
        plot_connectivity_matrix("blah", conn_idx, show_weight=False)

    with pytest.raises(TypeError, match="conn_idx must be an instance of"):
        plot_connectivity_matrix(net, "blah", show_weight=False)

    with pytest.raises(TypeError, match="show_weight must be an instance of"):
        plot_connectivity_matrix(net, conn_idx, show_weight="blah")

    src_gid = 5
    plot_cell_connectivity(net, conn_idx, src_gid, show=False)
    with pytest.raises(TypeError, match="net must be an instance of"):
        plot_cell_connectivity("blah", conn_idx, src_gid=src_gid)

    with pytest.raises(TypeError, match="conn_idx must be an instance of"):
        plot_cell_connectivity(net, "blah", src_gid)

    with pytest.raises(TypeError, match="src_gid must be an instance of"):
        plot_cell_connectivity(net, conn_idx, src_gid="blah")

    with pytest.raises(ValueError, match="src_gid -1 not a valid cell ID"):
        plot_cell_connectivity(net, conn_idx, src_gid=-1)

    # Test morphology plotting
    for cell_type in net.cell_types.values():
        cell_type.plot_morphology()
        cell_type.plot_morphology(color="r")

        sections = list(cell_type.sections.keys())
        section_color = {sect_name: f"C{idx}" for idx, sect_name in enumerate(sections)}
        cell_type.plot_morphology(color=section_color)

    cell_type = net.cell_types["L2_basket"]
    with pytest.raises(ValueError):
        cell_type.plot_morphology(color="z")
    with pytest.raises(ValueError):
        cell_type.plot_morphology(color={"soma": "z"})
    with pytest.raises(TypeError, match="color must be"):
        cell_type.plot_morphology(color=123)

    # test for invalid Axes object to plot_cells
    fig, axes = plt.subplots(1, 1)
    with pytest.raises(
        TypeError, match="'ax' to be an instance of Axes3D, but got Axes"
    ):
        plot_cells(net, ax=axes, show=False)
    cell_type.plot_morphology(pos=(1.0, 2.0, 3.0))
    with pytest.raises(TypeError, match="pos must be"):
        cell_type.plot_morphology(pos=123)
    with pytest.raises(ValueError, match="pos must be a tuple of 3 elements"):
        cell_type.plot_morphology(pos=(1, 2, 3, 4))
    with pytest.raises(TypeError, match="pos\\[idx\\] must be"):
        cell_type.plot_morphology(pos=(1, "2", 3))

    plt.close("all")

    # test interactive clicking updates the position of src_cell in plot
    del net.connectivity[-1]
    conn_idx = 15
    net.add_connection(
        net.gid_ranges["L2_pyramidal"][::2],
        "L5_basket",
        "soma",
        "ampa",
        0.00025,
        1.0,
        lamtha=3.0,
        probability=0.8,
    )
    fig = plot_cell_connectivity(net, conn_idx, show=False)
    ax_src, ax_target, _ = fig.axes

    pos = net.pos_dict["L2_pyramidal"][2]
    _fake_click(fig, ax_src, [pos[0], pos[1]])
    pos_in_plot = ax_target.collections[2].get_offsets().data[0]
    assert_allclose(pos[:2], pos_in_plot)
    plt.close("all")


def test_dipole_visualization(setup_net):
    """Test dipole visualisations."""
    net = setup_net

    # Test plotting of simulations with no spiking
    dpls = simulate_dipole(net, tstop=100.0, n_trials=1)
    net.cell_response.plot_spikes_raster()
    net.cell_response.plot_spikes_hist()

    weights_ampa = {"L2_pyramidal": 5.4e-5, "L5_pyramidal": 5.4e-5}
    syn_delays = {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0}

    net.add_bursty_drive(
        "beta_prox",
        tstart=0.0,
        burst_rate=25,
        burst_std=5,
        numspikes=1,
        spike_isi=0,
        n_drive_cells=11,
        location="proximal",
        weights_ampa=weights_ampa,
        synaptic_delays=syn_delays,
        event_seed=14,
    )

    net.add_bursty_drive(
        "beta_dist",
        tstart=0.0,
        burst_rate=25,
        burst_std=5,
        numspikes=1,
        spike_isi=0,
        n_drive_cells=11,
        location="distal",
        weights_ampa=weights_ampa,
        synaptic_delays=syn_delays,
        event_seed=14,
    )

    dpls = simulate_dipole(net, tstop=100.0, n_trials=2, record_vsec="all")
    fig = dpls[0].plot()  # plot the first dipole alone
    axes = fig.get_axes()[0]
    dpls[0].copy().smooth(window_len=10).plot(ax=axes)  # add smoothed versions
    dpls[0].copy().savgol_filter(h_freq=30).plot(ax=axes)  # on top

    # test decimation options
    plot_dipole(dpls[0], decim=2, show=False)
    for dec in [-1, [2, 2.0]]:
        with pytest.raises(
            ValueError, match="each decimation factor must be a positive"
        ):
            plot_dipole(dpls[0], decim=dec, show=False)

    # test plotting multiple dipoles as overlay
    fig = plot_dipole(dpls, show=False)

    # test plotting multiple dipoles with average
    fig = plot_dipole(dpls, average=True, show=False)
    plt.close("all")

    # test plotting dipoles with multiple layers
    fig, ax = plt.subplots()
    fig = plot_dipole(dpls, show=False, ax=[ax], layer=["L2"])
    fig = plot_dipole(dpls, show=False, layer=["L2", "L5", "agg"])
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig = plot_dipole(dpls, show=False, ax=axes, layer=["L2", "L5", "agg"])
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig = plot_dipole(
        dpls, show=False, ax=[axes[0], axes[1], axes[2]], layer=["L2", "L5", "agg"]
    )

    plt.close("all")

    with pytest.raises(AssertionError, match="ax and layer should have the same size"):
        fig, axes = plt.subplots(nrows=3, ncols=1)
        fig = plot_dipole(dpls, show=False, ax=axes, layer=["L2", "L5"])

    # multiple TFRs get averaged
    fig = plot_tfr_morlet(dpls, freqs=np.arange(23, 26, 1.0), n_cycles=3, show=False)
    # when min_freq > max_freq (y-axis inversion)
    fig = plot_tfr_morlet(dpls, freqs=np.array([30, 20, 10]), n_cycles=3, show=False)
    ax = fig.get_axes()[0]
    y_limits = ax.get_ylim()
    assert y_limits[0] > y_limits[1], (
        "Y-axis should be inverted when min_freq > max_freq"
    )

    with pytest.raises(RuntimeError, match="All dipoles must be scaled equally!"):
        plot_dipole([dpls[0].copy().scale(10), dpls[1].copy().scale(20)])
    with pytest.raises(RuntimeError, match="All dipoles must be scaled equally!"):
        plot_psd([dpls[0].copy().scale(10), dpls[1].copy().scale(20)])
    with pytest.raises(RuntimeError, match="All dipoles must be sampled equally!"):
        dpl_sfreq = dpls[0].copy()
        dpl_sfreq.sfreq /= 10
        plot_psd([dpls[0], dpl_sfreq])

    # pytest deprecation warning for tmin and tmax
    with pytest.deprecated_call():
        plot_dipole(dpls[0], show=False, tmin=10, tmax=100)

    # test cell response plotting
    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        net.cell_response.plot_spikes_raster(trial_idx="blah", show=False)
    net.cell_response.plot_spikes_raster(trial_idx=0, show=False)
    fig = net.cell_response.plot_spikes_raster(trial_idx=[0, 1], show=False)
    assert len(fig.axes[0].collections) > 0, "No data plotted in raster plot"

    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        net.cell_response.plot_spikes_hist(trial_idx="blah")
    net.cell_response.plot_spikes_hist(trial_idx=0, show=False)
    net.cell_response.plot_spikes_hist(trial_idx=[0, 1], show=False)
    net.cell_response.plot_spikes_hist(color="r")
    net.cell_response.plot_spikes_hist(color=["C0", "C1"])
    net.cell_response.plot_spikes_hist(color={"beta_prox": "r", "beta_dist": "g"})
    net.cell_response.plot_spikes_hist(
        spike_types={"group1": ["beta_prox", "beta_dist"]}, color={"group1": "r"}
    )
    net.cell_response.plot_spikes_hist(
        spike_types={"group1": ["beta"]}, color={"group1": "r"}
    )

    with pytest.raises(TypeError, match="color must be an instance of"):
        net.cell_response.plot_spikes_hist(color=123)
    with pytest.raises(ValueError):
        net.cell_response.plot_spikes_hist(color="z")
    with pytest.raises(ValueError):
        net.cell_response.plot_spikes_hist(color={"beta_prox": "z", "beta_dist": "g"})
    with pytest.raises(TypeError, match="Dictionary values of color must"):
        net.cell_response.plot_spikes_hist(color={"beta_prox": 123, "beta_dist": "g"})
    with pytest.raises(ValueError, match="'beta_dist' must be"):
        net.cell_response.plot_spikes_hist(color={"beta_prox": "r"})
    plt.close("all")


def test_drive_strength(setup_net):
    """Adds empty external drives to check there strength across each cell types"""
    net = setup_net

    weights_ampa = {"L2_pyramidal": 0.0, "L5_pyramidal": 0.0, "L2_basket": 0.0}
    synaptic_delays = {"L2_pyramidal": 0.0, "L5_pyramidal": 0.0, "L2_basket": 0.0}
    rate_constant = {"L2_pyramidal": 140.0, "L5_pyramidal": 40.0, "L2_basket": 100.0}

    net.add_poisson_drive(
        "poisson",
        rate_constant=rate_constant,
        weights_ampa=weights_ampa,
        location="proximal",
        synaptic_delays=synaptic_delays,
        event_seed=1349,
    )

    net.add_bursty_drive(
        "beta_dist",
        tstart=0.0,
        burst_rate=25,
        burst_std=5,
        numspikes=1,
        spike_isi=0,
        n_drive_cells=11,
        location="distal",
        weights_ampa=weights_ampa,
        synaptic_delays=synaptic_delays,
        event_seed=14,
    )

    figure = plot_drive_strength(net)

    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) > 0  # if there are any axes in the figure

    any_plot = any(ax.lines or ax.patches or ax.images for ax in figure.axes)
    assert any_plot  # At least one axis contains graphical elements

    plt.close("all")


class TestCellResponsePlotters:
    """Tests plotting methods of the CellResponse class"""

    @pytest.fixture(scope="class")
    def class_setup_net(self):
        """Creates a base network for tests within this class"""
        hnn_core_root = op.dirname(hnn_core.__file__)
        params_fname = op.join(hnn_core_root, "param", "default.json")
        params = read_params(params_fname)
        net = jones_2009_model(params, mesh_shape=(3, 3))

        return net

    @pytest.fixture(scope="class")
    def base_simulation_spikes(self, class_setup_net):
        """Adds drives with spikes for testing of spike visualizations"""
        net = class_setup_net
        weights_ampa = {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0}
        syn_delays = {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0}
        net.add_bursty_drive(
            "beta_prox",
            tstart=0.0,
            burst_rate=25,
            burst_std=5,
            numspikes=1,
            spike_isi=0,
            n_drive_cells=11,
            location="proximal",
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
            event_seed=14,
        )

        net.add_bursty_drive(
            "beta_dist",
            tstart=0.0,
            burst_rate=25,
            burst_std=5,
            numspikes=1,
            spike_isi=0,
            n_drive_cells=11,
            location="distal",
            weights_ampa=weights_ampa,
            synaptic_delays=syn_delays,
            event_seed=14,
        )
        dpls = simulate_dipole(net, tstop=100.0, n_trials=2, record_vsec="all")

        return net, dpls

    def test_spikes_raster_trial_idx(self, base_simulation_spikes):
        """Plotting with different index arguments"""
        net, _ = base_simulation_spikes

        # Bad index argument raises error
        with pytest.raises(TypeError, match="trial_idx must be an instance of"):
            net.cell_response.plot_spikes_raster(trial_idx="blah", show=False)

        # Test valid index arguments
        for index_arg in (0, [0, 1]):
            fig = net.cell_response.plot_spikes_raster(trial_idx=index_arg, show=False)
            # Check that collections contain data
            assert all(
                [
                    collection.get_positions() != [-1]
                    for collection in fig.axes[0].collections
                ]
            ), "No data plotted in raster plot"

    def test_spikes_raster_colors(self, base_simulation_spikes):
        """Plotting with different color arguments"""
        net, _ = base_simulation_spikes

        def _get_line_hex_colors(fig):
            colors = [
                matplotlib.colors.to_hex(line.get_color())
                for line in fig.axes[0].legend_.get_lines()
            ]
            labels = [text.get_text() for text in fig.axes[0].legend_.get_texts()]
            return colors, labels

        # Default colors should be the default color cycle
        fig = net.cell_response.plot_spikes_raster(trial_idx=0, show=False)
        colors, _ = _get_line_hex_colors(fig)
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][
            0 : len(colors)
        ]
        assert colors == default_colors

        # Custom hex colors as list
        custom_colors = ["#daf7a6", "#ffc300", "#ff5733", "#c70039"]
        fig = net.cell_response.plot_spikes_raster(
            trial_idx=0, show=False, colors=custom_colors
        )
        colors, _ = _get_line_hex_colors(fig)
        assert colors == custom_colors

        # Custom named colors as list
        custom_colors = ["skyblue", "maroon", "gold", "hotpink"]
        color_map = matplotlib.colors.get_named_colors_mapping()
        fig = net.cell_response.plot_spikes_raster(
            trial_idx=0, show=False, colors=custom_colors
        )
        colors, _ = _get_line_hex_colors(fig)
        assert colors == [color_map[color].lower() for color in custom_colors]

        # Incorrect number of colors as list
        too_few = ["r", "g", "b"]
        too_many = ["r", "g", "b", "y", "k"]
        for colors in [too_few, too_many]:
            with pytest.raises(ValueError, match="Number of colors must be equal to"):
                net.cell_response.plot_spikes_raster(
                    trial_idx=0, show=False, colors=colors
                )

        # Colors as dict mapping
        dict_mapping = {
            "L2_basket": "#daf7a6",
            "L2_pyramidal": "#ffc300",
            "L5_basket": "#ff5733",
            "L5_pyramidal": "#c70039",
        }
        fig = net.cell_response.plot_spikes_raster(
            trial_idx=0, show=False, colors=dict_mapping
        )
        colors, _ = _get_line_hex_colors(fig)
        assert colors == list(dict_mapping.values())

        # Change color of only one cell type
        dict_mapping = {"L2_pyramidal": "#daf7a6"}
        fig = net.cell_response.plot_spikes_raster(
            trial_idx=0, show=False, colors=dict_mapping
        )
        colors, cell_types = _get_line_hex_colors(fig)
        assert colors[cell_types.index("L2_pyramidal Spikes")] == "#daf7a6"

        # Invalid key in dict mapping
        dict_mapping = {"bad_cell_type": "#daf7a6"}
        with pytest.raises(ValueError, match="Invalid cell types provided."):
            net.cell_response.plot_spikes_raster(
                trial_idx=0, show=False, colors=dict_mapping
            )

    def test_spikes_raster_dipole_overlay(self, base_simulation_spikes):
        net, dpls = base_simulation_spikes

        # Missing dipole argument raises error
        # --------------------------------------------------
        with pytest.raises(ValueError, match="Dipole object must be provided"):
            net.cell_response.plot_spikes_raster(
                overlay_dipoles=True,
                dpl=None,
            )

        # Confirm dipoles are scaled correctly
        # --------------------------------------------------
        # Get initial y-axis ticks for raster plot without dipole
        fig = net.cell_response.plot_spikes_raster()
        initial_raster_yrange = fig.axes[0].get_yticks()

        # Get initial y-axis range, including a small allowance for a single
        # increment expansion on either end of the axis
        increment = abs(initial_raster_yrange[1] - initial_raster_yrange[0])
        initial_raster_yrange = (
            abs(max(initial_raster_yrange) - min(initial_raster_yrange)) + increment * 2
        )

        # Ensure dipoles are initially out of the bounds of the raster plot
        for dipole in dpls:
            for layer in ["L2", "L5"]:
                dipole.data[layer] = dipole.data[layer] * initial_raster_yrange

        # Get y-axis range of raster plot with overlaid dipoles
        fig = net.cell_response.plot_spikes_raster(
            overlay_dipoles=True,
            dpl=dpls,
        )
        updated_raster_yrange = fig.axes[0].get_yticks()
        updated_raster_yrange = abs(
            max(updated_raster_yrange) - min(updated_raster_yrange)
        )

        assert updated_raster_yrange <= initial_raster_yrange


def test_network_plotter_init(setup_net):
    """Test init keywords of NetworkPlotter class."""
    net = setup_net
    # test NetworkPlotter class
    args = [
        "xlim",
        "ylim",
        "zlim",
        "elev",
        "azim",
        "vmin",
        "vmax",
        "trial_idx",
        "time_idx",
        "colorbar",
    ]
    for arg in args:
        with pytest.raises(TypeError, match=f"{arg} must be"):
            net_plot = NetworkPlotter(net, **{arg: "blah"})

    net_plot = NetworkPlotter(net)

    assert net_plot.vsec_array.shape == (159, 1)
    assert net_plot.color_array.shape == (159, 1, 4)
    assert net_plot._vsec_recorded is False
    plt.close("all")


def test_network_plotter_simulation(setup_net):
    """Test NetworkPlotter class simulation warnings."""
    net = setup_net
    net_plot = NetworkPlotter(net)
    # Errors if vsec isn't recorded
    with pytest.raises(RuntimeError, match="Network must be simulated"):
        net_plot.export_movie("demo.gif", dpi=200)

    # Errors if vsec isn't recorded with record_vsec='all'
    _ = simulate_dipole(net, dt=0.5, tstop=10, record_vsec="soma")
    net_plot = NetworkPlotter(net)

    assert net_plot.vsec_array.shape == (159, 1)
    assert net_plot.color_array.shape == (159, 1, 4)
    assert net_plot._vsec_recorded is False

    with pytest.raises(RuntimeError, match="Network must be simulated"):
        net_plot.export_movie("demo.gif", dpi=200)

    net = setup_net
    _ = simulate_dipole(net, dt=0.5, tstop=10, record_vsec="all", n_trials=2)
    net_plot = NetworkPlotter(net)
    # setter/getter test for time_idx and trial_idx
    net_plot.time_idx = 5
    assert net_plot.time_idx == 5
    net_plot.trial_idx = 1
    assert net_plot.trial_idx == 1

    assert net_plot.vsec_array.shape == (159, 21)
    assert net_plot.color_array.shape == (159, 21, 4)
    assert net_plot._vsec_recorded is True
    assert isinstance(net_plot._cbar, Colorbar)
    plt.close("all")


def test_network_plotter_setter(setup_net):
    """Test NetworkPlotter class setters and getters."""
    net = setup_net
    net_plot = NetworkPlotter(net)
    # Type check errors
    args = [
        "xlim",
        "ylim",
        "zlim",
        "elev",
        "azim",
        "vmin",
        "vmax",
        "trial_idx",
        "time_idx",
        "colorbar",
    ]
    for arg in args:
        with pytest.raises(TypeError, match=f"{arg} must be"):
            setattr(net_plot, arg, "blah")

    # Check that the setters and getters work
    arg_dict = {
        "xlim": (-100, 100),
        "ylim": (-100, 100),
        "zlim": (-100, 100),
        "elev": 10,
        "azim": 10,
        "vmin": 0,
        "vmax": 100,
        "bgcolor": "white",
        "voltage_colormap": "jet",
        "colorbar": False,
    }
    for arg, val in arg_dict.items():
        setattr(net_plot, arg, val)
        assert getattr(net_plot, arg) == val

    assert net_plot._cbar is None
    assert net_plot.fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    # time_idx setter should raise an error if network is not simulated
    with pytest.raises(RuntimeError, match="Network must be simulated"):
        net_plot.time_idx = 5

    with pytest.raises(RuntimeError, match="Network must be simulated"):
        net_plot.trial_idx = 1
    plt.close("all")


def test_network_plotter_export(tmp_path, setup_net):
    """Test NetworkPlotter class export methods."""
    net = setup_net
    _ = simulate_dipole(net, dt=0.5, tstop=10, n_trials=1, record_vsec="all")
    net_plot = NetworkPlotter(net)

    # Check no file is already written
    path_out = tmp_path / "demo.gif"
    assert not path_out.is_file()

    # Test animation export and voltage plotting
    net_plot.export_movie(path_out, dpi=200, decim=100, writer="pillow")

    assert path_out.is_file()

    plt.close("all")


def test_invert_spike_types(setup_net):
    """Test plotting a histogram with an inverted external drive"""
    net = setup_net

    weights_ampa = {"L2_pyramidal": 0.15, "L5_pyramidal": 0.15}
    syn_delays = {"L2_pyramidal": 0.1, "L5_pyramidal": 1.0}

    net.add_evoked_drive(
        "evdist1",
        mu=63.53,
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="distal",
        synaptic_delays=syn_delays,
        event_seed=274,
    )

    net.add_evoked_drive(
        "evprox1",
        mu=26.61,
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa,
        location="proximal",
        synaptic_delays=syn_delays,
        event_seed=274,
    )

    _ = simulate_dipole(net, dt=0.5, tstop=80.0, n_trials=1)

    # test string input
    net.cell_response.plot_spikes_hist(
        spike_types=["evprox", "evdist"],
        invert_spike_types="evdist",
        show=False,
    )

    # test case where all inputs are flipped
    net.cell_response.plot_spikes_hist(
        spike_types=["evprox", "evdist"],
        invert_spike_types=["evprox", "evdist"],
        show=False,
    )

    # test case where some inputs are flipped
    fig = net.cell_response.plot_spikes_hist(
        spike_types=["evprox", "evdist"],
        invert_spike_types=["evdist"],
        show=False,
    )

    # check that there are 2 y axes
    assert len(fig.axes) == 2

    # check for equivalency of both y axes
    y1 = fig.axes[0]
    y2 = fig.axes[1]

    y1_max = max(y1.get_ylim())
    y2_max = max(y2.get_ylim())

    assert y1_max == y2_max

    # check that data are plotted
    assert y1_max > 1

    plt.close("all")
