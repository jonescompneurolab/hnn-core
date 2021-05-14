import os.path as op

import matplotlib
import numpy as np
import pytest

import hnn_core
from hnn_core import read_params, default_network
from hnn_core.viz import (plot_cells, plot_dipole, plot_psd, plot_tfr_morlet,
                          plot_cell_morphology)
from hnn_core.dipole import simulate_dipole

matplotlib.use('agg')


def test_network_visualization():
    """Test network visualisations."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3})
    net = default_network(params)
    plot_cells(net)
    with pytest.raises(ValueError, match='Unrecognized cell type'):
        plot_cell_morphology(cell_types='blah')
    axes = plot_cell_morphology(cell_types='L2Pyr')
    assert len(axes) == 1
    assert len(axes[0].lines) == 8


def test_dipole_visualization():
    """Test dipole visualisations."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 100.})
    net = default_network(params)
    weights_ampa_p = {'L2_pyramidal': 5.4e-5, 'L5_pyramidal': 5.4e-5}
    syn_delays_p = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}

    net.add_bursty_drive(
        'beta_prox', tstart=0., burst_rate=25, burst_std=5,
        numspikes=1, spike_isi=0, repeats=11, location='proximal',
        weights_ampa=weights_ampa_p, synaptic_delays=syn_delays_p, seedcore=14)

    dpls = simulate_dipole(net, n_trials=2, postproc=False)
    fig = dpls[0].plot()  # plot the first dipole alone
    axes = fig.get_axes()[0]
    dpls[0].copy().smooth(window_len=10).plot(ax=axes)  # add smoothed versions
    dpls[0].copy().savgol_filter(h_freq=30).plot(ax=axes)  # on top

    # test decimation options
    plot_dipole(dpls[0], decim=2)
    for dec in [-1, [2, 2.]]:
        with pytest.raises(ValueError,
                           match='each decimation factor must be a positive'):
            plot_dipole(dpls[0], decim=dec)

    # test plotting multiple dipoles as overlay
    fig = plot_dipole(dpls)

    # multiple TFRs get averaged
    fig = plot_tfr_morlet(dpls, freqs=np.arange(23, 26, 1.), n_cycles=3)

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
