from functools import partial
import os.path as op

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, jones_2009_model
from hnn_core.viz import plot_cells, plot_dipole, plot_psd, plot_tfr_morlet
from hnn_core.viz import plot_connectivity_matrix, plot_cell_connectivity
from hnn_core.dipole import simulate_dipole

matplotlib.use('agg')


def _fake_click(fig, ax, point, button=1):
    """Fake a click at a point within axes."""
    x, y = ax.transData.transform_point(point)
    func = partial(fig.canvas.button_press_event, x=x, y=y, button=button)
    func(guiEvent=None)


def test_network_visualization():
    """Test network visualisations."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3})
    net = jones_2009_model(params)
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

    cell_type = net.cell_response
    with pytest.raises(ValueError):
        cell_type.plot_morphology(color='z')
    with pytest.raises(TypeError, match='color must be'):
        cell_type.plot_morphology(color=123)

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


def test_dipole_visualization():
    """Test dipole visualisations."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3})
    net = jones_2009_model(params)
    weights_ampa_p = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
    syn_delays_p = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}

    net.add_bursty_drive(
        'beta_prox', tstart=0., burst_rate=25, burst_std=5,
        numspikes=1, spike_isi=0, n_drive_cells=11, location='proximal',
        weights_ampa=weights_ampa_p, synaptic_delays=syn_delays_p,
        event_seed=14)

    dpls = simulate_dipole(net, tstop=100., n_trials=2)
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

    # test cell response plotting
    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        net.cell_response.plot_spikes_raster(trial_idx='blah', show=False)
    net.cell_response.plot_spikes_raster(trial_idx=0, show=False)
    net.cell_response.plot_spikes_raster(trial_idx=[0, 1], show=False)

    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        net.cell_response.plot_spikes_hist(trial_idx='blah')
    net.cell_response.plot_spikes_hist(trial_idx=0, show=False)
    net.cell_response.plot_spikes_hist(trial_idx=[0, 1], show=False)

    plt.close('all')
