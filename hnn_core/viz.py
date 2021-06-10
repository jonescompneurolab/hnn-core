"""Visualization functions."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from itertools import cycle
from .externals.mne import _validate_type


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
    decim : int or list of int or None (default)
        Optional (integer) factor by which to decimate the raw dipole traces.
        The SciPy function :func:`~scipy.signal.decimate` is used, which
        recommends values <13. To achieve higher decimation factors, a list of
        ints can be provided. These are applied successively.
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
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    if isinstance(dpl, Dipole):
        dpl = [dpl]

    scale_applied = dpl[0].scale_applied
    for dpl_trial in dpl:
        if dpl_trial.scale_applied != scale_applied:
            raise RuntimeError('All dipoles must be scaled equally!')

        if layer in dpl_trial.data.keys():

            # extract scaled data and times
            data, times = _get_plot_data(dpl_trial, layer, tmin, tmax)
            if decim is not None:
                data, times = _decimate_plot_data(decim, data, times)

            ax.plot(times, data)

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
    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']
    cell_type_colors = {'L5_pyramidal': 'r', 'L5_basket': 'b',
                        'L2_pyramidal': 'g', 'L2_basket': 'w'}

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

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

    plt_show(show)
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

    for cell_type in net.cell_types:
        x = [pos[0] for pos in net.pos_dict[cell_type]]
        y = [pos[1] for pos in net.pos_dict[cell_type]]
        z = [pos[2] for pos in net.pos_dict[cell_type]]
        if cell_type in colors:
            color = colors[cell_type]
            marker = markers[cell_type]
            ax.scatter(x, y, z, c=color, marker=marker, label=cell_type)

    plt.legend(bbox_to_anchor=(-0.15, 1.025), loc="upper left")

    plt_show(show)
    return ax.get_figure()


def plot_tfr_morlet(dpl, freqs, *, n_cycles=7., tmin=None, tmax=None,
                    layer='agg', decim=None, padding='zeros', ax=None,
                    colormap='inferno', colorbar=True, show=True):
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

        data, times = _get_plot_data(dpl_trial, layer, tmin, tmax)

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
        cbar = fig.colorbar(im, ax=ax, format=xfmt, shrink=0.8, pad=0)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.set_ylabel(r'Power ([nAm $\times$ {:.0f}]$^2$)'.format(
            scale_applied), rotation=-90, va="bottom")

    plt_show(show)
    return ax.get_figure()


def plot_psd(dpl, *, fmin=0, fmax=None, tmin=None, tmax=None, layer='agg',
             ax=None, show=True):
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

        data, _ = _get_plot_data(dpl_trial, layer, tmin, tmax)

        freqs, Pxx = periodogram(data, sfreq, window='hamming', nfft=len(data))
        trial_power.append(Pxx)

    ax.plot(freqs, np.mean(np.array(Pxx, ndmin=2), axis=0))
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

    if ax is None:
        plt.figure()
        ax = plt.axes(projection='3d')

    # Cell is in XZ plane
    ax.set_xlim((cell.pos[1] - 250, cell.pos[1] + 150))
    ax.set_zlim((cell.pos[2] - 100, cell.pos[2] + 1200))

    for sec_name, p_sec in cell.p_secs.items():
        linewidth = _linewidth_from_data_units(ax, p_sec['diam'])
        sec_pts = p_sec['sec_pts']
        xs, ys, zs = list(), list(), list()
        for pt in sec_pts:
            dx = cell.pos[0] - cell.p_secs['soma']['sec_pts'][0][0]
            dy = cell.pos[1] - cell.p_secs['soma']['sec_pts'][0][1]
            dz = cell.pos[2] - cell.p_secs['soma']['sec_pts'][0][2]
            xs.append(pt[0] + dx)
            ys.append(pt[1] + dz)
            zs.append(pt[2] + dy)
        ax.plot(xs, ys, zs, 'b-', linewidth=linewidth)
    ax.view_init(0, -90)
    ax.axis('off')

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

    src_range = np.array(conn['src_range'])
    target_range = np.array(conn['target_range'])
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
                    src_pos, target_pos, nc_dict)
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


def plot_cell_connectivity(net, conn_idx, src_gid, ax=None, colorbar=True,
                           colormap='viridis', show=True):
    """Plot synaptic weight of connections originating from src_gid.

    Parameters
    ----------
    net : Instance of Network object
        The Network object
    conn_idx : int
        Index of connection to be visualized
        from `net.connectivity`
    src_gid : int
        Each cell in a network is uniquely identified by it's "global ID": GID.
    ax : instance of Axes3D
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
    Target cells will be determined by the connection class given by
    net.connectivity[conn_idx].
    If the target cell is not connected to src_gid, it will appear as a
    smaller black circle.
    src_gid is plotted as a red circle. src_gid will not be plotted if
    the connection corresponds to a drive, ex: poisson, bursty, etc.

    """
    import matplotlib.pyplot as plt
    from .network import Network
    from .cell import _get_gaussian_connection
    from matplotlib.ticker import ScalarFormatter

    _validate_type(net, Network, 'net', 'Network')
    _validate_type(conn_idx, int, 'conn_idx', 'int')
    _validate_type(src_gid, int, 'src_gid', 'int')
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Load objects for distance calculation
    conn = net.connectivity[conn_idx]
    nc_dict = conn['nc_dict']
    src_type = conn['src_type']
    target_type = conn['target_type']
    src_type_pos = net.pos_dict[src_type]
    target_type_pos = net.pos_dict[target_type]

    src_range = np.array(conn['src_range'])
    if src_gid not in src_range:
        raise ValueError(f'src_gid not in the src type range of {src_type} '
                         f'gids. Valid gids include {src_range[0]} -> '
                         f'{src_range[-1]}')

    target_range = np.array(conn['target_range'])

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
        weight, _ = _get_gaussian_connection(src_pos, target_pos, nc_dict)
        weights.append(weight)

    im = ax.scatter(target_x_pos, target_y_pos, c=weights, s=50, cmap=colormap)

    # Gather positions of all gids in target_type.
    x_pos = [target_type_pos[idx][0] for idx in range(len(target_type_pos))]
    y_pos = [target_type_pos[idx][1] for idx in range(len(target_type_pos))]
    ax.scatter(x_pos, y_pos, color='k', marker='x', zorder=-1, s=20)

    # Only plot src_gid if proper cell type.
    if src_type in net.cell_types:
        ax.scatter(src_pos[0], src_pos[1], marker='s', color='red', s=150)
    ax.set_ylabel('Y Position')
    ax.set_xlabel('X Position')
    ax.set_title(f"{conn['src_type']}-> {conn['target_type']}"
                 f" ({conn['loc']}, {conn['receptor']})")

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        cbar = fig.colorbar(im, ax=ax, format=xfmt)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.set_ylabel('Weight', rotation=-90, va="bottom")

    plt.tight_layout()
    plt_show(show)
    return ax.get_figure(), ax
