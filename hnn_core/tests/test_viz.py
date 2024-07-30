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
from hnn_core.viz import (plot_cells, plot_dipole, plot_psd, plot_tfr_morlet,
                          plot_connectivity_matrix, plot_cell_connectivity,
                          NetworkPlotter)
from hnn_core.dipole import simulate_dipole

matplotlib.use('agg')


@pytest.fixture
def setup_net():
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = jones_2009_model(params, mesh_shape=(3, 3))

    return net


def _fake_click(fig, ax, point, button=1):
    """Fake a click at a point within axes."""
    x, y = ax.transData.transform_point(point)
    button_press_event = backend_bases.MouseEvent(
        name='button_press_event', canvas=fig.canvas,
        x=x, y=y, button=button
    )
    fig.canvas.callbacks.process('button_press_event', button_press_event)


def test_network_visualization(setup_net):
    """Test network visualisations."""
    net = setup_net
    plot_cells(net)
    ax = net.cell_types['L2_pyramidal'].plot_morphology()
    assert len(ax.lines) == 8

    conn_idx = 0
    plot_connectivity_matrix(net, conn_idx, show=False)
    with pytest.raises(TypeError, match='net must be an instance of'):
        plot_connectivity_matrix('blah', conn_idx, show_weight=False)

    with pytest.raises(TypeError, match='conn_idx must be an instance of'):
        plot_connectivity_matrix(net, 'blah', show_weight=False)

    with pytest.raises(TypeError, match='show_weight must be an instance of'):
        plot_connectivity_matrix(net, conn_idx, show_weight='blah')

    src_gid = 5
    plot_cell_connectivity(net, conn_idx, src_gid, show=False)
    with pytest.raises(TypeError, match='net must be an instance of'):
        plot_cell_connectivity('blah', conn_idx, src_gid=src_gid)

    with pytest.raises(TypeError, match='conn_idx must be an instance of'):
        plot_cell_connectivity(net, 'blah', src_gid)

    with pytest.raises(TypeError, match='src_gid must be an instance of'):
        plot_cell_connectivity(net, conn_idx, src_gid='blah')

    with pytest.raises(ValueError, match='src_gid -1 not a valid cell ID'):
        plot_cell_connectivity(net, conn_idx, src_gid=-1)

    # Test morphology plotting
    for cell_type in net.cell_types.values():
        cell_type.plot_morphology()
        cell_type.plot_morphology(color='r')

        sections = list(cell_type.sections.keys())
        section_color = {sect_name: f'C{idx}' for
                         idx, sect_name in enumerate(sections)}
        cell_type.plot_morphology(color=section_color)

    cell_type = net.cell_types['L2_basket']
    with pytest.raises(ValueError):
        cell_type.plot_morphology(color='z')
    with pytest.raises(ValueError):
        cell_type.plot_morphology(color={'soma': 'z'})
    with pytest.raises(TypeError, match='color must be'):
        cell_type.plot_morphology(color=123)

    # test for invalid Axes object to plot_cells
    fig, axes = plt.subplots(1, 1)
    with pytest.raises(TypeError,
                       match="'ax' to be an instance of Axes3D, but got Axes"):
        plot_cells(net, ax=axes, show=False)
    cell_type.plot_morphology(pos=(1.0, 2.0, 3.0))
    with pytest.raises(TypeError, match='pos must be'):
        cell_type.plot_morphology(pos=123)
    with pytest.raises(ValueError, match='pos must be a tuple of 3 elements'):
        cell_type.plot_morphology(pos=(1, 2, 3, 4))
    with pytest.raises(TypeError, match='pos\\[idx\\] must be'):
        cell_type.plot_morphology(pos=(1, '2', 3))

    plt.close('all')

    # test interactive clicking updates the position of src_cell in plot
    del net.connectivity[-1]
    conn_idx = 15
    net.add_connection(net.gid_ranges['L2_pyramidal'][::2],
                       'L5_basket', 'soma',
                       'ampa', 0.00025, 1.0, lamtha=3.0,
                       probability=0.8)
    fig = plot_cell_connectivity(net, conn_idx, show=False)
    ax_src, ax_target, _ = fig.axes

    pos = net.pos_dict['L2_pyramidal'][2]
    _fake_click(fig, ax_src, [pos[0], pos[1]])
    pos_in_plot = ax_target.collections[2].get_offsets().data[0]
    assert_allclose(pos[:2], pos_in_plot)
    plt.close('all')


def test_dipole_visualization(setup_net):
    """Test dipole visualisations."""
    net = setup_net

    # Test plotting of simulations with no spiking
    dpls = simulate_dipole(net, tstop=100., n_trials=1)
    net.cell_response.plot_spikes_raster()
    net.cell_response.plot_spikes_hist()

    weights_ampa = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
    syn_delays = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}

    net.add_bursty_drive(
        'beta_prox', tstart=0., burst_rate=25, burst_std=5,
        numspikes=1, spike_isi=0, n_drive_cells=11, location='proximal',
        weights_ampa=weights_ampa, synaptic_delays=syn_delays,
        event_seed=14)

    net.add_bursty_drive(
        'beta_dist', tstart=0., burst_rate=25, burst_std=5,
        numspikes=1, spike_isi=0, n_drive_cells=11, location='distal',
        weights_ampa=weights_ampa, synaptic_delays=syn_delays,
        event_seed=14)

    dpls = simulate_dipole(net, tstop=100., n_trials=2, record_vsec='all')
    fig = dpls[0].plot()  # plot the first dipole alone
    axes = fig.get_axes()[0]
    dpls[0].copy().smooth(window_len=10).plot(ax=axes)  # add smoothed versions
    dpls[0].copy().savgol_filter(h_freq=30).plot(ax=axes)  # on top

    # test decimation options
    plot_dipole(dpls[0], decim=2, show=False)
    for dec in [-1, [2, 2.]]:
        with pytest.raises(ValueError,
                           match='each decimation factor must be a positive'):
            plot_dipole(dpls[0], decim=dec, show=False)

    # test plotting multiple dipoles as overlay
    fig = plot_dipole(dpls, show=False)

    # test plotting multiple dipoles with average
    fig = plot_dipole(dpls, average=True, show=False)
    plt.close('all')

    # test plotting dipoles with multiple layers
    fig, ax = plt.subplots()
    fig = plot_dipole(dpls, show=False, ax=[ax], layer=['L2'])
    fig = plot_dipole(dpls, show=False, layer=['L2', 'L5', 'agg'])
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig = plot_dipole(dpls, show=False, ax=axes, layer=['L2', 'L5', 'agg'])
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig = plot_dipole(dpls,
                      show=False,
                      ax=[axes[0], axes[1], axes[2]],
                      layer=['L2', 'L5', 'agg'])

    plt.close('all')

    with pytest.raises(AssertionError,
                       match="ax and layer should have the same size"):
        fig, axes = plt.subplots(nrows=3, ncols=1)
        fig = plot_dipole(dpls, show=False, ax=axes, layer=['L2', 'L5'])

    # multiple TFRs get averaged
    fig = plot_tfr_morlet(dpls, freqs=np.arange(23, 26, 1.), n_cycles=3,
                          show=False)

    with pytest.raises(RuntimeError,
                       match="All dipoles must be scaled equally!"):
        plot_dipole([dpls[0].copy().scale(10), dpls[1].copy().scale(20)])
    with pytest.raises(RuntimeError,
                       match="All dipoles must be scaled equally!"):
        plot_psd([dpls[0].copy().scale(10), dpls[1].copy().scale(20)])
    with pytest.raises(RuntimeError,
                       match="All dipoles must be sampled equally!"):
        dpl_sfreq = dpls[0].copy()
        dpl_sfreq.sfreq /= 10
        plot_psd([dpls[0], dpl_sfreq])

    # pytest deprecation warning for tmin and tmax
    with pytest.deprecated_call():
        plot_dipole(dpls[0], show=False, tmin=10, tmax=100)

    # test cell response plotting
    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        net.cell_response.plot_spikes_raster(trial_idx='blah', show=False)
    net.cell_response.plot_spikes_raster(trial_idx=0, show=False)
    fig = net.cell_response.plot_spikes_raster(trial_idx=[0, 1], show=False)
    assert len(fig.axes[0].collections) > 0, "No data plotted in raster plot"

    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        net.cell_response.plot_spikes_hist(trial_idx='blah')
    net.cell_response.plot_spikes_hist(trial_idx=0, show=False)
    net.cell_response.plot_spikes_hist(trial_idx=[0, 1], show=False)
    net.cell_response.plot_spikes_hist(color='r')
    net.cell_response.plot_spikes_hist(color=['C0', 'C1'])
    net.cell_response.plot_spikes_hist(color={'beta_prox': 'r',
                                              'beta_dist': 'g'})
    net.cell_response.plot_spikes_hist(
        spike_types={'group1': ['beta_prox', 'beta_dist']},
        color={'group1': 'r'})
    net.cell_response.plot_spikes_hist(
        spike_types={'group1': ['beta']}, color={'group1': 'r'})

    with pytest.raises(TypeError, match="color must be an instance of"):
        net.cell_response.plot_spikes_hist(color=123)
    with pytest.raises(ValueError):
        net.cell_response.plot_spikes_hist(color='z')
    with pytest.raises(ValueError):
        net.cell_response.plot_spikes_hist(color={'beta_prox': 'z',
                                                  'beta_dist': 'g'})
    with pytest.raises(TypeError, match="Dictionary values of color must"):
        net.cell_response.plot_spikes_hist(color={'beta_prox': 123,
                                                  'beta_dist': 'g'})
    with pytest.raises(ValueError, match="'beta_dist' must be"):
        net.cell_response.plot_spikes_hist(color={'beta_prox': 'r'})
    plt.close('all')


def test_network_plotter_init(setup_net):
    """Test init keywords of NetworkPlotter class."""
    net = setup_net
    # test NetworkPlotter class
    args = ['xlim', 'ylim', 'zlim', 'elev', 'azim', 'vmin', 'vmax',
            'trial_idx', 'time_idx', 'colorbar']
    for arg in args:
        with pytest.raises(TypeError, match=f'{arg} must be'):
            net_plot = NetworkPlotter(net, **{arg: 'blah'})

    net_plot = NetworkPlotter(net)

    assert net_plot.vsec_array.shape == (159, 1)
    assert net_plot.color_array.shape == (159, 1, 4)
    assert net_plot._vsec_recorded is False
    plt.close('all')


def test_network_plotter_simulation(setup_net):
    """Test NetworkPlotter class simulation warnings."""
    net = setup_net
    net_plot = NetworkPlotter(net)
    # Errors if vsec isn't recorded
    with pytest.raises(RuntimeError, match='Network must be simulated'):
        net_plot.export_movie('demo.gif', dpi=200)

    # Errors if vsec isn't recorded with record_vsec='all'
    _ = simulate_dipole(net, dt=0.5, tstop=10, record_vsec='soma')
    net_plot = NetworkPlotter(net)

    assert net_plot.vsec_array.shape == (159, 1)
    assert net_plot.color_array.shape == (159, 1, 4)
    assert net_plot._vsec_recorded is False

    with pytest.raises(RuntimeError, match='Network must be simulated'):
        net_plot.export_movie('demo.gif', dpi=200)

    net = setup_net
    _ = simulate_dipole(net, dt=0.5, tstop=10, record_vsec='all', n_trials=2)
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
    plt.close('all')


def test_network_plotter_setter(setup_net):
    """Test NetworkPlotter class setters and getters."""
    net = setup_net
    net_plot = NetworkPlotter(net)
    # Type check errors
    args = ['xlim', 'ylim', 'zlim', 'elev', 'azim', 'vmin', 'vmax',
            'trial_idx', 'time_idx', 'colorbar']
    for arg in args:
        with pytest.raises(TypeError, match=f'{arg} must be'):
            setattr(net_plot, arg, 'blah')

    # Check that the setters and getters work
    arg_dict = {'xlim': (-100, 100), 'ylim': (-100, 100), 'zlim': (-100, 100),
                'elev': 10, 'azim': 10, 'vmin': 0, 'vmax': 100,
                'bgcolor': 'white', 'voltage_colormap': 'jet',
                'colorbar': False}
    for arg, val in arg_dict.items():
        setattr(net_plot, arg, val)
        assert getattr(net_plot, arg) == val

    assert net_plot._cbar is None
    assert net_plot.fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    # time_idx setter should raise an error if network is not simulated
    with pytest.raises(RuntimeError, match='Network must be simulated'):
        net_plot.time_idx = 5

    with pytest.raises(RuntimeError, match='Network must be simulated'):
        net_plot.trial_idx = 1
    plt.close('all')


def test_network_plotter_export(tmp_path, setup_net):
    """Test NetworkPlotter class export methods."""
    net = setup_net
    _ = simulate_dipole(net, dt=0.5, tstop=10, n_trials=1,
                        record_vsec='all')
    net_plot = NetworkPlotter(net)

    # Check no file is already written
    path_out = tmp_path / 'demo.gif'
    assert not path_out.is_file()

    # Test animation export and voltage plotting
    net_plot.export_movie(path_out, dpi=200, decim=100, writer='pillow')

    assert path_out.is_file()

    plt.close('all')
