"""Visualization functions."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from itertools import cycle


def _get_plot_data(dpl, layer, tmin, tmax):
    plot_tmin = dpl.times[0]
    if tmin is not None:
        plot_tmin = max(tmin, plot_tmin)
    plot_tmax = dpl.times[-1]
    if tmax is not None:
        plot_tmax = min(tmax, plot_tmax)

    mask = np.logical_and(dpl.times >= plot_tmin, dpl.times < plot_tmax)
    times = dpl.times[mask]
    data = dpl.data[layer][mask]

    return data, times


def _decimate_plot_data(decim, data, times):
    from scipy.signal import decimate
    data = decimate(data, decim)
    times = times[::decim]
    return data, times


def plot_dipole(dpl, tmin=None, tmax=None, ax=None, layer='agg', decim=None,
                show=True):
    """Simple layer-specific plot function.

    Parameters
    ----------
    dpl : instance of Dipole | list of Dipole instances
        The Dipole object.
    tmin : float or None
        Start time of plot in milliseconds. If None, plot entire simulation.
    tmax : float or None
        End time of plot in milliseconds. If None, plot entire simulation.
    ax : instance of matplotlib figure | None
        The matplotlib axis
    layer : str
        The layer to plot. Can be one of
        'agg', 'L2', and 'L5'
    decimate : int
        Factor by which to decimate the raw dipole traces (optional)
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of plt.fig
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from .dipole import Dipole

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if isinstance(dpl, Dipole):
        dpl = [dpl]

    for dpl_trial in dpl:
        if layer in dpl_trial.data.keys():

            data, times = _get_plot_data(dpl_trial, layer, tmin, tmax)
            if decim is not None:
                data, times = _decimate_plot_data(decim, data, times)

            ax.plot(times, data)

    ax.ticklabel_format(axis='both', scilimits=(-2, 3))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Dipole moment')
    if layer == 'agg':
        title_str = 'Aggregate (L2 + L5)'
    else:
        title_str = layer
    ax.set_title(title_str)

    if show:
        plt.show()
    return ax.get_figure()


def plot_spikes_hist(cell_response, ax=None, spike_types=None, show=True):
    """Plot the histogram of spiking activity across trials.

    Parameters
    ----------
    cell_response : instance of CellResponse
        The CellResponse object from net.cell_response
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None,
        a new figure is created.
    spike_types: string | list | dictionary | None
        String input of a valid spike type is plotted individually.
            Ex: 'poisson', 'evdist', 'evprox', ...
        List of valid string inputs will plot each spike type individually.
            Ex: ['poisson', 'evdist']
        Dictionary of valid lists will plot list elements as a group.
            Ex: {'Evoked': ['evdist', 'evprox'], 'Tonic': ['poisson']}
        If None, all input spike types are plotted individually if any
        are present. Otherwise spikes from all cells are plotted.
        Valid strings also include leading characters of spike types
            Example: 'ev' is equivalent to ['evdist', 'evprox']
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    spike_times = np.array(sum(cell_response._spike_times, []))
    spike_types_data = np.array(sum(cell_response._spike_types, []))

    unique_types = np.unique(spike_types_data)
    spike_types_mask = {s_type: np.in1d(spike_types_data, s_type)
                        for s_type in unique_types}
    cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
    input_types = np.setdiff1d(unique_types, cell_types)

    if isinstance(spike_types, str):
        spike_types = {spike_types: [spike_types]}

    if spike_types is None:
        if any(input_types):
            spike_types = input_types.tolist()
        else:
            spike_types = unique_types.tolist()
    if isinstance(spike_types, list):
        spike_types = {s_type: [s_type] for s_type in spike_types}
    if isinstance(spike_types, dict):
        for spike_label in spike_types:
            if not isinstance(spike_types[spike_label], list):
                raise TypeError(f'spike_types[{spike_label}] must be a list. '
                                f'Got '
                                f'{type(spike_types[spike_label]).__name__}.')

    if not isinstance(spike_types, dict):
        raise TypeError('spike_types should be str, list, dict, or None')

    spike_labels = dict()
    for spike_label, spike_type_list in spike_types.items():
        for spike_type in spike_type_list:
            n_found = 0
            for unique_type in unique_types:
                if unique_type.startswith(spike_type):
                    if unique_type in spike_labels:
                        raise ValueError(f'Elements of spike_types must map to'
                                         f' mutually exclusive input types.'
                                         f' {unique_type} is found more than'
                                         f' once.')
                    spike_labels[unique_type] = spike_label
                    n_found += 1
            if n_found == 0:
                raise ValueError(f'No input types found for {spike_type}')

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    color_cycle = cycle(['r', 'g', 'b', 'y', 'm', 'c'])

    bins = np.linspace(0, spike_times[-1], 50)
    spike_color = dict()
    for spike_type, spike_label in spike_labels.items():
        label = "_nolegend_"
        if spike_label not in spike_color:
            spike_color[spike_label] = next(color_cycle)
            label = spike_label

        color = spike_color[spike_label]
        ax.hist(spike_times[spike_types_mask[spike_type]], bins,
                label=label, color=color)
    plt.legend()

    if show:
        plt.show()
    return ax.get_figure()


def plot_spikes_raster(cell_response, ax=None, show=True):
    """Plot the aggregate spiking activity according to cell type.

    Parameters
    ----------
    cell_response : instance of CellResponse
        The CellResponse object from net.cell_response
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None,
        a new figure is created.
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure object.
    """

    import matplotlib.pyplot as plt
    spike_times = np.array(sum(cell_response._spike_times, []))
    spike_types = np.array(sum(cell_response._spike_types, []))
    spike_gids = np.array(sum(cell_response._spike_gids, []))
    cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
    cell_type_colors = {'L5_pyramidal': 'r', 'L5_basket': 'b',
                        'L2_pyramidal': 'g', 'L2_basket': 'w'}

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ypos = 0
    for cell_type in cell_types:
        cell_type_gids = np.unique(spike_gids[spike_types == cell_type])
        cell_type_times, cell_type_ypos = [], []
        for gid in cell_type_gids:
            gid_time = spike_times[spike_gids == gid]
            cell_type_times.append(gid_time)
            cell_type_ypos.append(np.repeat(ypos, len(gid_time)))
            ypos = ypos - 1

        if cell_type_times:
            cell_type_times = np.concatenate(cell_type_times)
            cell_type_ypos = np.concatenate(cell_type_ypos)
        else:
            cell_type_times = []
            cell_type_ypos = []

        ax.scatter(cell_type_times, cell_type_ypos, label=cell_type,
                   color=cell_type_colors[cell_type])

    ax.legend(loc=1)
    ax.set_facecolor('k')
    ax.set_xlabel('Time (ms)')
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(left=0)

    if show:
        plt.show()
    return ax.get_figure()


def plot_cells(net, ax=None, show=True):
    """Plot the cells using Network.pos_dict.

    Parameters
    ----------
    net : instance of NetworkBuilder
        The NetworkBuilder object.
    ax : instance of matplotlib Axes3D | None
        An axis object from matplotlib. If None,
        a new figure is created.
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    colors = {'L5_pyramidal': 'b', 'L2_pyramidal': 'c',
              'L5_basket': 'r', 'L2_basket': 'm'}
    markers = {'L5_pyramidal': '^', 'L2_pyramidal': '^',
               'L5_basket': 'x', 'L2_basket': 'x'}

    for cell_type in net.cellname_list:
        x = [pos[0] for pos in net.pos_dict[cell_type]]
        y = [pos[1] for pos in net.pos_dict[cell_type]]
        z = [pos[2] for pos in net.pos_dict[cell_type]]
        if cell_type in colors:
            color = colors[cell_type]
            marker = markers[cell_type]
            ax.scatter(x, y, z, c=color, marker=marker, label=cell_type)

    plt.legend(bbox_to_anchor=(-0.15, 1.025), loc="upper left")

    if show:
        plt.show()

    return ax.get_figure()


def plot_tfr_morlet(dpl, *, freqs, n_cycles=7., tmin=None, tmax=None,
                    layer='agg', decim=False, ax=None,
                    colorbar=True, show=True):
    """Plot Morlet time-frequency representation of dipole time course

    Parameters
    ----------
    dpl : instance of Dipole | list of Dipole instances
        The Dipole object.
    fregs : array
        Frequency range of interest.
    n_cycles : float | array of float, default 7.0
        Number of cycles. Fixed number or one per frequency.
    tmin : float or None
        Start time of plot in milliseconds. If None, plot entire simulation.
    tmax : float or None
        End time of plot in milliseconds. If None, plot entire simulation.
    layer : str, default 'agg'
        The layer to plot. Can be one of 'agg', 'L2', and 'L5'
    decim : int or None
        Factor by which to decimate the raw dipole traces (optional)
    ax : instance of matplotlib figure | None
        The matplotlib axis
    colorbar : bool
        If True (default), adjust figure to include colorbar.
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from mne.time_frequency import tfr_array_morlet

    data, times = _get_plot_data(dpl, layer, tmin, tmax)

    sfreq = dpl.sfreq
    if decim is not None:
        data, times = _decimate_plot_data(decim, data, times)
        sfreq = sfreq / decim

    # mirror padding!
    data = np.r_[data[-1:0:-1], data, data[-2::-1]]

    # MNE expects an array of shape (n_trials, n_channels, n_times)
    data = data[None, None, :]
    power = tfr_array_morlet(data, sfreq=sfreq, freqs=freqs,
                             n_cycles=n_cycles, output='power')

    # get the middle portion after mirroring
    power = power[:, :, :, times.shape[0] - 1:2 * times.shape[0] - 1]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    im = ax.pcolormesh(times, freqs, power[0, 0, ...], cmap='inferno',
                       shading='auto')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')

    if colorbar:
        fig = ax.get_figure()
        fig.subplots_adjust(right=0.8)
        l, b, w, h = ax.get_position().bounds
        cb_h = 0.8 * h
        cb_b = b + (h - cb_h) / 2
        cbar_ax = fig.add_axes([l + w + 0.05, cb_b, 0.03, cb_h])
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        fig.colorbar(im, cax=cbar_ax, format=xfmt)

    if show:
        plt.show()
    return ax.get_figure()


def plot_spectrogram(dpl, *, fmin, fmax, winlen=None, tmin=None, tmax=None,
                     layer='agg', ax=None, show=True):
    """Plot Welch spectrogram (power spectrum) of dipole time course

    Parameters
    ----------
    dpl : instance of Dipole | list of Dipole instances
        The Dipole object.
    fmin : float
        Minimum frequency to plot (in Hz).
    fmax : float
        Maximum frequency to plot (in Hz).
    winlen : float | None
        Length of window (in ms) to average using Welch periodogram method.
        The actual window size used depends on the sampling rate (closest
        power of 2, rounded up). If None, entire window is used (no averaging).
    tmin : float or None
        Start time of data to include (in ms). If None, use entire simulation.
    tmax : float or None
        End time of data to include (in ms). If None, use entire simulation.
    layer : str, default 'agg'
        The layer to plot. Can be one of 'agg', 'L2', and 'L5'
    ax : instance of matplotlib figure | None
        The matplotlib axis.
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    sfreq = dpl.sfreq
    data, times = _get_plot_data(dpl, layer, tmin, tmax)

    if winlen is None:
        winlen = times[-1] - times[0]
    nfft = 1e-3 * winlen * sfreq
    nperseg = 2 ** int(np.ceil(np.log2(nfft)))

    freqs, _, psds = spectrogram(data, sfreq, window='hamming',
                                 nperseg=nperseg, noverlap=0)
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(freqs, np.mean(psds, axis=-1))
    ax.set_xlim((fmin, fmax))
    ax.ticklabel_format(axis='both', scilimits=(-2, 3))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density')
    if show:
        plt.show()
    return ax.get_figure()
