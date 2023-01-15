"""Visualization functions."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

import numpy as np
from itertools import cycle
import colorsys

from .externals.mne import _validate_type


def _lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def _get_plot_data_trange(times, data, tmin, tmax):
    """Get slices of times and data based on tmin and tmax"""
    if isinstance(times, list):
        times = np.array(times)
    if isinstance(data, list):
        data = np.array(data)
    plot_tmin = times[0]
    if tmin is not None:
        plot_tmin = max(tmin, plot_tmin)
    plot_tmax = times[-1]
    if tmax is not None:
        plot_tmax = min(tmax, plot_tmax)

    mask = np.logical_and(times >= plot_tmin, times < plot_tmax)

    return data[mask], times[mask]


def _decimate_plot_data(decim, data, times, sfreq=None):
    from scipy.signal import decimate
    if not isinstance(decim, list):
        decim = [decim]

    for dec in decim:
        if not isinstance(dec, int) or dec < 1:
            raise ValueError('each decimation factor must be a positive int, '
                             f'but {dec} is a {type(dec)}')
        data = decimate(data, dec)
        times = times[::dec]

    if sfreq is None:
        return data, times
    else:
        sfreq /= np.prod(decim)
        return data, times, sfreq


def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    NB copied from :func:`mne.viz.utils.plt_show`.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    if show and get_backend() != 'agg':
        (fig or plt).show(**kwargs)


def plot_laminar_lfp(times, data, contact_labels, tmin=None, tmax=None,
                     ax=None, decim=None, color='cividis',
                     voltage_offset=50, voltage_scalebar=200, show=True):
    """Plot laminar extracellular electrode array voltage time series.

    Parameters
    ----------
    times : array-like, shape (n_times,)
        Sampling times (in ms).
    data : Two-dimensional Numpy array
        The extracellular voltages as an (n_contacts, n_times) array.
    tmin : float | None
        Start time of plot in milliseconds. If None, plot entire simulation.
    tmax : float | None
        End time of plot in milliseconds. If None, plot entire simulation.
    ax : instance of matplotlib figure | None
        The matplotlib axis
    decim : int | list of int | None (default)
        Optional (integer) factor by which to decimate the raw dipole traces.
        The SciPy function :func:`~scipy.signal.decimate` is used, which
        recommends values <13. To achieve higher decimation factors, a list of
        ints can be provided. These are applied successively.
    color : str | array of floats | ``matplotlib.colors.ListedColormap``
        The colormap to use for plotting. The usual Matplotlib standard
        colormap strings may be used (e.g., 'jetblue'). A color can also be
        defined as an RGBA-quadruplet, or an array of RGBA-values (one for each
        electrode contact trace to plot). An instance of
        :class:`~matplotlib.colors.ListedColormap` may also be provided.
    voltage_offset : float | None (optional)
        Amount to offset traces by on the voltage-axis. Useful for plotting
        laminar arrays.
    voltage_scalebar : float | None (optional)
        Height, in units of uV, of a scale bar to plot in the top-left corner
        of the plot.
    contact_labels : list
        Labels associated with the contacts to plot. Passed as-is to
        :func:`~matplotlib.axes.Axes.set_yticklabels`.
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of plt.fig
        The matplotlib figure handle into which time series were plotted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    _validate_type(times, (list, np.ndarray), 'times')
    _validate_type(data, (list, np.ndarray), 'data')
    if isinstance(times, list):
        times = np.array(times)
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError(f'data must be 2D, got shape {data.shape}')
    if len(times) != data.shape[1]:
        raise ValueError(f'length of times ({len(times)}) and data '
                         f'({len(data)}) do not match')

    n_contacts = data.shape[0]
    if color is not None:
        _validate_type(color,
                       (str, tuple, list, np.ndarray, ListedColormap),
                       'color')
        if isinstance(color, (tuple, list)):
            if (not np.all([isinstance(c, float) for c in color]) or
                    len(color) < 3 or len(color) > 4):
                raise ValueError(
                    f'color must be length 3 or 4, got {color}')
        elif isinstance(color, np.ndarray):
            if (color.shape[0] != n_contacts or
                    (color.shape[1] < 3 or color.shape[1] > 4)):
                raise ValueError(
                    f'color must be n_contacts x (3 or 4), got {color}')
        elif isinstance(color, ListedColormap):
            if color.N != n_contacts:
                raise ValueError(f'ListedColormap has N={color.N}, but '
                                 f'there are {n_contacts} contacts')
        elif isinstance(color, str):
            color = plt.get_cmap(color, len(contact_labels))

    if ax is None:
        _, ax = plt.subplots(1, 1)

    n_offsets = data.shape[0]
    trace_offsets = np.zeros((n_offsets, 1))
    if voltage_offset is not None:
        trace_offsets = np.arange(n_offsets)[:, np.newaxis] * voltage_offset

    for contact_no, trace in enumerate(np.atleast_2d(data)):
        plot_data, plot_times = _get_plot_data_trange(times, trace, tmin, tmax)

        if decim is not None:
            plot_data, plot_times = _decimate_plot_data(decim, plot_data,
                                                        plot_times)

        if isinstance(color, np.ndarray):
            col = color[contact_no]
        elif isinstance(color, ListedColormap):
            col = color(contact_no)
        else:
            col = color
        ax.plot(plot_times, plot_data + trace_offsets[contact_no],
                label=f'C{contact_no}', color=col)

    if voltage_offset is not None:
        ax.set_ylim(-voltage_offset, n_offsets * voltage_offset)
        ylabel = 'Individual contact traces'
        if len(contact_labels) != n_offsets:
            raise ValueError(f'contact_labels is length {len(contact_labels)},'
                             f' but {n_offsets} contacts to be plotted')
        else:
            trace_ticks = np.arange(0, len(contact_labels) * voltage_offset,
                                    voltage_offset)
            ax.set_yticks(trace_ticks)
            ax.set_yticklabels(contact_labels)

        if voltage_scalebar is None:
            voltage_scalebar = voltage_offset

    if voltage_scalebar is not None:
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        scalebar = AnchoredSizeBar(ax.transData, 1,
                                   f'{voltage_scalebar:.0f} ' + r'$\mu V$',
                                   'upper left',
                                   size_vertical=voltage_scalebar,
                                   pad=0.1,
                                   color='black',
                                   label_top=False,
                                   frameon=False)
        ax.add_artist(scalebar)
    else:
        ylabel = r'Electric potential ($\mu V$)'
        ax.ticklabel_format(axis='both', scilimits=(-2, 3))

    ax.set_ylabel(ylabel, multialignment='center')
    ax.set_xlabel('Time (ms)')

    plt_show(show)
    return ax.get_figure()


def plot_dipole(dpl, tmin=None, tmax=None, ax=None, layer='agg', decim=None,
                color='k', label="average", average=False, show=True):
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
    decim : int or list of int or None (default)
        Optional (integer) factor by which to decimate the raw dipole traces.
        The SciPy function :func:`~scipy.signal.decimate` is used, which
        recommends values <13. To achieve higher decimation factors, a list of
        ints can be provided. These are applied successively.
    color : tuple of float | str
        RGBA value to use for plotting. By default, 'k' (black)
    label : str
        Dipole label. Enabled when average=True
    average : bool
        If True, render the average across all dpls.
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of plt.fig
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from .dipole import Dipole, average_dipoles

    layers = layer if isinstance(layer, list) else [layer]
    if ax is None:
        _, ax = plt.subplots(len(layers),
                             1,
                             constrained_layout=True,
                             sharex=True,
                             sharey=True)
    axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

    if isinstance(dpl, Dipole):
        dpl = [dpl]
    elif average:
        dpl = dpl + [average_dipoles(dpl)]

    scale_applied = dpl[0].scale_applied

    assert len(layers) == len(axes), "ax and layer should have the same size"

    for layer, ax in zip(layers, axes):
        for idx, dpl_trial in enumerate(dpl):
            if dpl_trial.scale_applied != scale_applied:
                raise RuntimeError('All dipoles must be scaled equally!')

            if layer in dpl_trial.data.keys():

                # extract scaled data and times
                data, times = _get_plot_data_trange(dpl_trial.times,
                                                    dpl_trial.data[layer],
                                                    tmin, tmax)
                if decim is not None:
                    data, times = _decimate_plot_data(decim, data, times)
                if idx == len(dpl) - 1 and average:
                    # the average dpl
                    ax.plot(times, data, color=color, label=label, lw=1.5)
                else:
                    alpha = 0.5 if average else 1.
                    ax.plot(times, data, color=_lighten_color(color, 0.5),
                            alpha=alpha, lw=1.)

        if average:
            ax.legend()

        ax.ticklabel_format(axis='both', scilimits=(-2, 3))
        ax.set_xlabel('Time (ms)')
        if scale_applied == 1:
            ylabel = 'Dipole moment (nAm)'
        else:
            ylabel = 'Dipole moment\n(nAm ' +\
                r'$\times$ {:.0f})'.format(scale_applied)
        ax.set_ylabel(ylabel, multialignment='center')
        if layer == 'agg':
            title_str = 'Aggregate (L2 + L5)'
        else:
            title_str = layer
        ax.set_title(title_str)

    plt_show(show)
    return axes[0].get_figure()


def plot_spikes_hist(cell_response, trial_idx=None, ax=None, spike_types=None,
                     show=True):
    """Plot the histogram of spiking activity across trials.

    Parameters
    ----------
    cell_response : instance of CellResponse
        The CellResponse object from net.cell_response
    trial_idx : int | list of int | None
        Index of trials to be plotted. If None, all trials plotted.
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None,
        a new figure is created.
    spike_types: string | list | dictionary | None
        String input of a valid spike type is plotted individually.

        | Ex: ``'poisson'``, ``'evdist'``, ``'evprox'``, ...

        List of valid string inputs will plot each spike type individually.

        | Ex: ``['poisson', 'evdist']``

        Dictionary of valid lists will plot list elements as a group.

        | Ex: ``{'Evoked': ['evdist', 'evprox'], 'Tonic': ['poisson']}``

        If None, all input spike types are plotted individually if any
        are present. Otherwise spikes from all cells are plotted.
        Valid strings also include leading characters of spike types

        | Ex: ``'ev'`` is equivalent to ``['evdist', 'evprox']``
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    n_trials = len(cell_response.spike_times)
    if trial_idx is None:
        trial_idx = list(range(n_trials))

    if isinstance(trial_idx, int):
        trial_idx = [trial_idx]
    _validate_type(trial_idx, list, 'trial_idx', 'int, list of int')

    # Extract desired trials
    if len(cell_response._spike_times[0]) > 0:
        spike_times = np.concatenate(
            np.array(cell_response._spike_times)[trial_idx])
        spike_types_data = np.concatenate(
            np.array(cell_response._spike_types)[trial_idx])
    else:
        spike_times = np.array([])
        spike_types_data = np.array([])

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
        _, ax = plt.subplots(1, 1, constrained_layout=True)

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
    ax.set_ylabel("Counts")
    ax.legend()

    plt_show(show)
    return ax.get_figure()


def plot_spikes_raster(cell_response, trial_idx=None, ax=None, show=True):
    """Plot the aggregate spiking activity according to cell type.

    Parameters
    ----------
    cell_response : instance of CellResponse
        The CellResponse object from net.cell_response
    trial_idx : int | list of int | None
        Index of trials to be plotted. If None, all trials plotted
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None, a new figure is created.
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure object.
    """

    import matplotlib.pyplot as plt
    n_trials = len(cell_response.spike_times)
    if trial_idx is None:
        trial_idx = list(range(n_trials))

    if isinstance(trial_idx, int):
        trial_idx = [trial_idx]
    _validate_type(trial_idx, list, 'trial_idx', 'int, list of int')

    # Extract desired trials
    if len(cell_response._spike_times[0]) > 0:
        spike_times = np.concatenate(
            np.array(cell_response._spike_times)[trial_idx])
        spike_types = np.concatenate(
            np.array(cell_response._spike_types)[trial_idx])
        spike_gids = np.concatenate(
            np.array(cell_response._spike_gids)[trial_idx])
    else:
        spike_times = np.array([])
        spike_types = np.array([])
        spike_gids = np.array([])

    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']
    cell_type_colors = {'L5_pyramidal': 'r', 'L5_basket': 'b',
                        'L2_pyramidal': 'g', 'L2_basket': 'w'}

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    ypos = 0
    events = []
    for cell_type in cell_types:
        cell_type_gids = np.unique(spike_gids[spike_types == cell_type])
        cell_type_times, cell_type_ypos = [], []
        for gid in cell_type_gids:
            gid_time = spike_times[spike_gids == gid]
            cell_type_times.append(gid_time)
            cell_type_ypos.append(ypos)
            ypos = ypos - 1

        if cell_type_times:
            events.append(
                ax.eventplot(cell_type_times, lineoffsets=cell_type_ypos,
                             color=cell_type_colors[cell_type],
                             label=cell_type, linelengths=5))

    ax.legend(handles=[e[0] for e in events], loc=1)
    ax.set_facecolor('k')
    ax.set_xlabel('Time (ms)')
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(left=0)

    plt_show(show)
    return ax.get_figure()


def plot_cells(net, ax=None, show=True):
    """Plot the cells using Network.pos_dict.

    Parameters
    ----------
    net : instance of Network
        The Network object.
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

    for cell_type in net.cell_types:
        x = [pos[0] for pos in net.pos_dict[cell_type]]
        y = [pos[1] for pos in net.pos_dict[cell_type]]
        z = [pos[2] for pos in net.pos_dict[cell_type]]
        if cell_type in colors:
            color = colors[cell_type]
            marker = markers[cell_type]
            ax.scatter(x, y, z, c=color, s=50, marker=marker, label=cell_type)

    if net.rec_arrays:
        cols = plt.get_cmap('inferno', len(net.rec_arrays) + 2)
        for ii, (arr_name, arr) in enumerate(net.rec_arrays.items()):
            x = [p[0] for p in arr.positions]
            y = [p[1] for p in arr.positions]
            z = [p[2] for p in arr.positions]
            ax.scatter(x, y, z, color=cols(ii + 1), s=25, marker='o',
                       label=arr_name)

    plt.legend(bbox_to_anchor=(-0.15, 1.025), loc="upper left")

    plt_show(show)
    return ax.get_figure()


def plot_tfr_morlet(dpl, freqs, *, n_cycles=7., tmin=None, tmax=None,
                    layer='agg', decim=None, padding='zeros', ax=None,
                    colormap='inferno', colorbar=True, colorbar_inside=False,
                    show=True):
    """Plot Morlet time-frequency representation of dipole time course

    Parameters
    ----------
    dpl : instance of Dipole | list of Dipole instances
        The Dipole object. If a list of dipoles is given, the power is
        calculated separately for each trial, then averaged.
    freqs : array
        Frequency range of interest.
    n_cycles : float or array of float, default 7.0
        Number of cycles. Fixed number or one per frequency.
    tmin : float or None
        Start time of plot in milliseconds. If None, plot entire simulation.
    tmax : float or None
        End time of plot in milliseconds. If None, plot entire simulation.
    layer : str, default 'agg'
        The layer to plot. Can be one of 'agg', 'L2', and 'L5'
    decim : int or list of int or None (default)
        Optional (integer) factor by which to decimate the raw dipole traces.
        The SciPy function :func:`~scipy.signal.decimate` is used, which
        recommends values <13. To achieve higher decimation factors, a list of
        ints can be provided. These are applied successively.
    padding : str or None
        Optional padding of the dipole time course beyond the plotting limits.
        Possible values are: 'zeros' for padding with 0's (default), 'mirror'
        for mirror-image padding.
    ax : instance of matplotlib figure | None
        The matplotlib axis
    colormap : str
        The name of a matplotlib colormap, e.g., 'viridis'. Default: 'inferno'
    colorbar : bool
        If True (default), adjust figure to include colorbar.
    colorbar_inside: bool, default False
        Put the color inside the heatmap if True.
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from .externals.mne import tfr_array_morlet
    from .dipole import Dipole

    if isinstance(dpl, Dipole):
        dpl = [dpl]

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    scale_applied = dpl[0].scale_applied
    sfreq = dpl[0].sfreq
    trial_power = []
    for dpl_trial in dpl:
        if dpl_trial.scale_applied != scale_applied:
            raise RuntimeError('All dipoles must be scaled equally!')
        if dpl_trial.sfreq != sfreq:
            raise RuntimeError('All dipoles must be sampled equally!')

        data, times = _get_plot_data_trange(dpl_trial.times,
                                            dpl_trial.data[layer],
                                            tmin, tmax)

        sfreq = dpl_trial.sfreq
        if decim is not None:
            data, times, sfreq = _decimate_plot_data(decim, data, times,
                                                     sfreq=sfreq)

        if padding is not None:
            if not isinstance(padding, str):
                raise ValueError('padding must be a string (or None)')
            if padding == 'zeros':
                data = np.r_[np.zeros((len(data) - 1,)), data.ravel(),
                             np.zeros((len(data) - 1,))]
            elif padding == 'mirror':
                data = np.r_[data[-1:0:-1], data, data[-2::-1]]

        # MNE expects an array of shape (n_trials, n_channels, n_times)
        data = data[None, None, :]
        power = tfr_array_morlet(data, sfreq=sfreq, freqs=freqs,
                                 n_cycles=n_cycles, output='power')

        if padding is not None:
            # get the middle portion after padding
            power = power[:, :, :, times.shape[0] - 1:2 * times.shape[0] - 1]
        trial_power.append(power)

    power = np.mean(trial_power, axis=0)
    im = ax.pcolormesh(times, freqs, power[0, 0, ...], cmap=colormap,
                       shading='auto')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        # default colorbar
        if colorbar_inside is False:
            cbar = fig.colorbar(im, ax=ax, format=xfmt, shrink=0.8, pad=0)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.set_ylabel(r'Power ([nAm $\times$ {:.0f}]$^2$)'.format(
                scale_applied), rotation=-90, va="bottom")
        # put colorbar inside the heatmap.
        else:
            cbar_color = "white"
            cbar_fontsize = 6

            ax_pos = ax.get_position()
            ax_width = ax_pos.x1 - ax_pos.x0
            ax_height = ax_pos.y1 - ax_pos.y0
            cbar_L = ax_pos.x0 + 0.9 * ax_width
            cbar_B = ax_pos.y0 + 0.8 * ax_height
            cbar_W = ax_width * 0.04
            cbar_H = ax_height * 0.15

            cax = fig.add_axes([cbar_L, cbar_B, cbar_W, cbar_H])
            cbar = fig.colorbar(im, cax=cax, format=xfmt, shrink=0.8, pad=0)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.offsetText.set_fontsize(cbar_fontsize)

            cbar.ax.set_ylabel(
                r'Power ([nAm $\times$ {:.0f}]$^2$)'.format(scale_applied),
                rotation=-90, va="bottom", fontsize=cbar_fontsize,
                color=cbar_color)
            cbar.ax.tick_params(direction='in', labelsize=cbar_fontsize,
                                labelcolor=cbar_color, colors=cbar_color)
            plt.setp(cbar.ax.spines.values(), color=cbar_color)
            setattr(fig, f'_cbar-ax-{id(ax)}', cbar)

    plt_show(show)
    return ax.get_figure()


def plot_psd(dpl, *, fmin=0, fmax=None, tmin=None, tmax=None, layer='agg',
             color=None, label=None, ax=None, show=True):
    """Plot power spectral density (PSD) of dipole time course

    Applies `~scipy.signal.periodogram` from SciPy with ``window='hamming'``.
    Note that no spectral averaging is applied across time, as most
    ``hnn_core`` simulations are short-duration. However, passing a list of
    `Dipole` instances will plot their average (Hamming-windowed) power, which
    resembles the `Welch`-method applied over time.

    Parameters
    ----------
    dpl : instance of Dipole | list of Dipole instances
        The Dipole object.
    fmin : float
        Minimum frequency to plot (in Hz). Default: 0 Hz
    fmax : float
        Maximum frequency to plot (in Hz). Default: None (plot up to Nyquist)
    tmin : float or None
        Start time of data to include (in ms). If None, use entire simulation.
    tmax : float or None
        End time of data to include (in ms). If None, use entire simulation.
    layer : str, default 'agg'
        The layer to plot. Can be one of 'agg', 'L2', and 'L5'
    color : str or tuple or None
        The line color of PSD
    label : str or None
        Line label for PSD
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
    from scipy.signal import periodogram
    from .dipole import Dipole

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    if isinstance(dpl, Dipole):
        dpl = [dpl]

    scale_applied = dpl[0].scale_applied
    sfreq = dpl[0].sfreq
    trial_power = []
    for dpl_trial in dpl:
        if dpl_trial.scale_applied != scale_applied:
            raise RuntimeError('All dipoles must be scaled equally!')
        if dpl_trial.sfreq != sfreq:
            raise RuntimeError('All dipoles must be sampled equally!')

        data, _ = _get_plot_data_trange(dpl_trial.times,
                                        dpl_trial.data[layer],
                                        tmin, tmax)

        freqs, Pxx = periodogram(data, sfreq, window='hamming', nfft=len(data))
        trial_power.append(Pxx)

    ax.plot(freqs, np.mean(np.array(Pxx, ndmin=2), axis=0), color=color,
            label=label)
    if label:
        ax.legend()
    if fmax is not None:
        ax.set_xlim((fmin, fmax))
    ax.ticklabel_format(axis='both', scilimits=(-2, 3))
    ax.set_xlabel('Frequency (Hz)')
    if scale_applied == 1:
        ylabel = 'Power spectral density\n(nAm' + r'$^2 \ Hz^{-1}$)'
    else:
        ylabel = 'Power spectral density\n' +\
            r'([nAm$\times$ {:.0f}]'.format(scale_applied) +\
            r'$^2 \ Hz^{-1}$)'
    ax.set_ylabel(ylabel, multialignment='center')

    plt_show(show)
    return ax.get_figure()


def _linewidth_from_data_units(ax, linewidth):
    # see: https://stackoverflow.com/a/35501485
    fig = ax.get_figure()
    length = fig.bbox_inches.width * ax.get_position().width
    value_range = np.diff(ax.get_xlim())[0]
    length *= 72  # Convert length to points
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def plot_cell_morphology(cell, ax, show=True):
    """Plot the cell morphology.

    Parameters
    ----------
    cell : instance of Cell
        The cell object
    ax : instance of Axes3D
        Matplotlib 3D axis
    show : bool
        If True, show the plot

    Returns
    -------
    axes : list of instance of Axes3D
        The matplotlib 3D axis handle.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    cell_list = list()
    colors = ['b', 'c', 'r', 'm']
    clr_index = 0

    if ax is None:
        plt.figure()
        ax = plt.axes(projection='3d')

    if type(cell) is dict:
        for ind_cell in cell:
            cell_list = list(cell.values())
    else:
        cell_list[0] = cell

    # Cell is in XZ plane
    # ax.set_xlim((cell_list[0].pos[1] - 250, cell_list[0].pos[1] + 150))
    # ax.set_zlim((cell_list[0].pos[2] - 100, cell_list[0].pos[2] + 1200))
    cell_radii = list()
    cell_radii.append(clr_index)
    for clr_index, cell in enumerate(cell_list):

        # Calculating the radius for cell offset
        radius = 0
        for sec_name, section in cell.sections.items():
            end_pts = section.end_pts
            xs, ys, zs = list(), list(), list()
            for pt in end_pts:
                dx = cell.pos[0] - cell.sections['soma'].end_pts[0][0]
                dy = cell.pos[1] - cell.sections['soma'].end_pts[0][1]
                dz = cell.pos[2] - cell.sections['soma'].end_pts[0][2]
                if radius < pt[0]:
                    radius = pt[0]
        cell_radii.append(radius)

        # Plotting the cell
        for sec_name, section in cell.sections.items():
            ax.set_xlim((sum(cell_radii, 100)))
            ax.set_zlim((cell.pos[2] - 100, cell.pos[2] + 1200))
            linewidth = _linewidth_from_data_units(ax, section.diam)
            end_pts = section.end_pts
            xs, ys, zs = list(), list(), list()
            for pt in end_pts:
                dx = cell.pos[0] - cell.sections['soma'].end_pts[0][0]
                dy = cell.pos[1] - cell.sections['soma'].end_pts[0][1]
                dz = cell.pos[2] - cell.sections['soma'].end_pts[0][2]
                xs.append(pt[0] + dx + (radius + cell_radii[-1] + 100))
                ys.append(pt[1] + dz)
                zs.append(pt[2] + dy)
            ax.plot(xs, ys, zs, color=colors[clr_index], linewidth=linewidth)
        ax.view_init(0, -90)
        ax.axis('on')
        ax.grid('off')
    plt.tight_layout()
    plt_show(show)
    return ax


def plot_connectivity_matrix(net, conn_idx, ax=None, show_weight=True,
                             colorbar=True, colormap='Greys',
                             show=True):
    """Plot connectivity matrix with color bar for synaptic weights

    Parameters
    ----------
    net : Instance of Network object
        The Network object
    conn_idx : int
        Index of connection to be visualized
        from `net.connectivity`
    ax : instance of Axes3D
        Matplotlib 3D axis
    show_weight : bool
        If True, visualize connectivity weights as gradient.
        If False, all weights set to constant value.
    colormap : str
        The name of a matplotlib colormap. Default: 'Greys'
    colorbar : bool
        If True (default), adjust figure to include colorbar.
    show : bool
        If True, show the plot

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from .network import Network
    from .cell import _get_gaussian_connection

    _validate_type(net, Network, 'net', 'Network')
    _validate_type(conn_idx, int, 'conn_idx', 'int')
    _validate_type(show_weight, bool, 'show_weight', 'bool')
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Load objects for distance calculation
    conn = net.connectivity[conn_idx]
    nc_dict = conn['nc_dict']
    src_type = conn['src_type']
    target_type = conn['target_type']
    src_type_pos = net.pos_dict[src_type]
    target_type_pos = net.pos_dict[target_type]

    src_range = np.array(net.gid_ranges[conn['src_type']])
    target_range = np.array(net.gid_ranges[conn['target_type']])
    connectivity_matrix = np.zeros((len(src_range), len(target_range)))

    for src_gid, target_src_pair in conn['gid_pairs'].items():
        src_idx = np.where(src_range == src_gid)[0][0]
        target_indeces = np.where(np.in1d(target_range, target_src_pair))[0]
        for target_idx in target_indeces:
            src_pos = src_type_pos[src_idx]
            target_pos = target_type_pos[target_idx]

            # Identical calculation used in Cell.par_connect_from_src()
            if show_weight:
                weight, _ = _get_gaussian_connection(
                    src_pos, target_pos, nc_dict,
                    inplane_distance=net._inplane_distance)
            else:
                weight = 1.0

            connectivity_matrix[src_idx, target_idx] = weight

    im = ax.imshow(connectivity_matrix, cmap=colormap, interpolation='none')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        cbar = fig.colorbar(im, ax=ax, format=xfmt)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.set_ylabel('Weight', rotation=-90, va="bottom")

    ax.set_xlabel(f"{conn['target_type']} target gids "
                  f"({target_range[0]}-{target_range[-1]})")
    ax.set_xticklabels(list())
    ax.set_ylabel(f"{conn['src_type']} source gids "
                  f"({src_range[0]}-{src_range[-1]})")
    ax.set_yticklabels(list())
    ax.set_title(f"{conn['src_type']} -> {conn['target_type']} "
                 f"({conn['loc']}, {conn['receptor']})")

    plt.tight_layout()
    plt_show(show)
    return ax.get_figure()


def _update_target_plot(ax, conn, src_gid, src_type_pos, target_type_pos,
                        src_range, target_range, nc_dict, colormap,
                        inplane_distance):
    from .cell import _get_gaussian_connection

    # Extract indeces to get position in network
    # Index in gid range aligns with net.pos_dict
    target_src_pair = conn['gid_pairs'][src_gid]
    target_indeces = np.where(np.in1d(target_range, target_src_pair))[0]

    src_idx = np.where(src_range == src_gid)[0][0]
    src_pos = src_type_pos[src_idx]

    # Aggregate positions and weight of each connected target
    weights, target_x_pos, target_y_pos = list(), list(), list()
    for target_idx in target_indeces:
        target_pos = target_type_pos[target_idx]
        target_x_pos.append(target_pos[0])
        target_y_pos.append(target_pos[1])
        weight, _ = _get_gaussian_connection(src_pos, target_pos, nc_dict,
                                             inplane_distance)
        weights.append(weight)

    ax.clear()
    im = ax.scatter(target_x_pos, target_y_pos, c=weights, s=50,
                    cmap=colormap)
    x_pos = target_type_pos[:, 0]
    y_pos = target_type_pos[:, 1]
    ax.scatter(x_pos, y_pos, color='k', marker='x', zorder=-1, s=20)
    ax.scatter(src_pos[0], src_pos[1], marker='s', color='red', s=150)
    ax.set_ylabel('Y Position')
    ax.set_xlabel('X Position')
    return im


def plot_cell_connectivity(net, conn_idx, src_gid=None, axes=None,
                           colorbar=True, colormap='viridis', show=True):
    """Plot synaptic weight of connections.

    This is an interactive plot with source cells shown in the left
    subplot and connectivity from a source cell to all the target cells
    in the right subplot. Click on the cells in the left subplot to
    explore how the connectivity pattern changes for different source cells.

    Parameters
    ----------
    net : Instance of Network object
        The Network object
    conn_idx : int
        Index of connection to be visualized from net.connectivity
    src_gid : int | None
        The cell ID of the source cell. It must be an element of
        net.connectivity[conn_idx]['gid_pairs'].keys()
        If None, the first cell from the list of valid src_gids is selected.
    axes : instance of Axes3D
        Matplotlib 3D axis
    colormap : str
        The name of a matplotlib colormap. Default: 'viridis'
    colorbar : bool
        If True (default), adjust figure to include colorbar.
    show : bool
        If True, show the plot

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.

    Notes
    -----
    Target cells will be determined by the connections in
    net.connectivity[conn_idx].
    If the target cell is not connected to the source cell,
    it will appear as a smaller black cross.
    Source cell is plotted as a red square. Source cell will not be plotted if
    the connection corresponds to a drive, ex: poisson, bursty, etc.
    """
    import matplotlib.pyplot as plt
    from .network import Network
    from matplotlib.ticker import ScalarFormatter

    _validate_type(net, Network, 'net', 'Network')
    _validate_type(conn_idx, int, 'conn_idx', 'int')

    # Load objects for distance calculation
    conn = net.connectivity[conn_idx]
    nc_dict = conn['nc_dict']
    src_type = conn['src_type']
    target_type = conn['target_type']
    src_type_pos = np.array(net.pos_dict[src_type])
    target_type_pos = np.array(net.pos_dict[target_type])
    src_range = np.array(net.gid_ranges[conn['src_type']])

    valid_src_gids = list(net.connectivity[conn_idx]['gid_pairs'].keys())
    src_pos_valid = src_type_pos[np.in1d(src_range, valid_src_gids)]

    if src_gid is None:
        src_gid = valid_src_gids[0]
    _validate_type(src_gid, int, 'src_gid', 'int')

    if src_gid not in valid_src_gids:
        raise ValueError(f'src_gid {src_gid} not a valid cell ID for this '
                         f'connection. Please select one of {valid_src_gids}')

    target_range = np.array(net.gid_ranges[conn['target_type']])

    if axes is None:
        if src_type in net.cell_types:
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        else:
            fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
            axes = [axes]

    if len(axes) == 2:
        ax_src, ax = axes
    else:
        ax = axes[0]

    im = _update_target_plot(ax, conn, src_gid, src_type_pos,
                             target_type_pos, src_range,
                             target_range, nc_dict, colormap,
                             net._inplane_distance)

    x_src = src_type_pos[:, 0]
    y_src = src_type_pos[:, 1]
    x_src_valid = src_pos_valid[:, 0]
    y_src_valid = src_pos_valid[:, 1]
    if src_type in net.cell_types:
        ax_src.scatter(x_src, y_src, marker='s', color='red', s=50,
                       alpha=0.2)
        ax_src.scatter(x_src_valid, y_src_valid, marker='s', color='red',
                       s=50)

    plt.suptitle(f"{conn['src_type']}-> {conn['target_type']}"
                 f" ({conn['loc']}, {conn['receptor']})")

    def _onclick(event):
        if event.inaxes in [ax] or event.inaxes is None:
            return

        dist = np.linalg.norm(src_type_pos[:, :2] -
                              np.array([event.xdata, event.ydata]),
                              axis=1)
        src_idx = np.argmin(dist)

        src_gid = src_range[src_idx]
        if src_gid not in valid_src_gids:
            return
        _update_target_plot(ax, conn, src_gid, src_type_pos,
                            target_type_pos, src_range, target_range,
                            nc_dict, colormap, net._inplane_distance)

        fig.canvas.draw()

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        cbar = fig.colorbar(im, ax=ax, format=xfmt)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.set_ylabel('Weight', rotation=-90, va="bottom")

    plt.tight_layout()

    fig.canvas.mpl_connect('button_press_event', _onclick)

    plt_show(show)
    return ax.get_figure()


def plot_laminar_csd(times, data, contact_labels, ax=None, colorbar=True,
                     show=True):
    """Plot laminar current source density (CSD) estimation from LFP array.

    Parameters
    ----------
    times : Numpy array, shape (n_times,)
        Sampling times (in ms).
    data : array-like, shape (n_channels, n_times)
        CSD data, channels x time.
    ax : instance of matplotlib figure | None
        The matplotlib axis.
    colorbar : bool
        If the colorbar is presented.
    contact_labels : list
        Labels associated with the contacts to plot. Passed as-is to
        :func:`~matplotlib.axes.Axes.set_yticklabels`.
    show : bool
        If True, show the plot.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    im = ax.pcolormesh(times, contact_labels, np.array(data),
                       cmap="jet_r", shading='auto')
    ax.set_title("CSD")

    if colorbar:
        color_axis = ax.inset_axes([1.05, 0, 0.02, 1], transform=ax.transAxes)
        plt.colorbar(im, ax=ax, cax=color_axis).set_label(r'$CSD (uV/um^{2})$')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Electrode depth')
    plt.tight_layout()
    plt_show(show)

    return ax.get_figure()
