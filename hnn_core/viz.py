"""Visualization functions."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from cycler import cycler


def plot_dipole(dpl, ax=None, layer='agg', show=True):
    """Simple layer-specific plot function.

    Parameters
    ----------
    dpl : instance of Dipole | list of Dipole instances
        The Dipole object.
    ax : instance of matplotlib figure | None
        The matplotlib axis
    layer : str
        The layer to plot. Can be one of
        'agg', 'L2', and 'L5'
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
            ax.plot(dpl_trial.times, dpl_trial.data[layer])

    ax.set_xlabel('Time (ms)')
    ax.set_title(layer)

    if show:
        plt.show()
    return ax.get_figure()


def plot_hist_input(spikes, ax=None, spike_types=None, show=True):
    """Plot the histogram of input.

    Parameters
    ----------
    spikes : instance of Spikes
        The spikes object from net.spikes
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None,
        a new figure is created.
    spike_types: string | list | dictionary | None
        String input of a valid spike type is plotted individually.
            Ex: 'common', 'evdist', 'evprox', 'extgauss', 'extpois'
        List of valid string inputs will plot each spike type individually.
            Ex: ['common', 'evdist']
        Dictionary of valid lists will plot list elements as a group.
            Ex: {'Evoked': ['evdist', 'evprox'], 'External': ['extpois']}
        If None, all default spike types are plotted individually.
            Default: ['common', 'evdist', 'evprox', 'extgauss', 'extpois']
        Valid strings also include leading characters of spike types
            Example: 'ext' is equivalent to ['extgauss', 'extpois']
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    spike_times = np.array(sum(spikes._times, []))
    spike_types_data = np.array(sum(spikes._types, []))
    default_types = ['evprox', 'evdist', 'common', 'extgauss', 'extpois']
    cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
    input_types = np.setdiff1d(np.unique(spike_types_data), cell_types)
    spike_types_mask = {s_type: np.in1d(spike_types_data, s_type)
                        for s_type in input_types}

    if isinstance(spike_types, str):
        spike_types = {spike_types: [spike_types]}
    elif isinstance(spike_types, list):
        spike_types = {s_type: [s_type] for s_type in spike_types}
    elif spike_types is None:
        spike_types = {s_type: [s_type] for s_type in default_types}
    elif not isinstance(spike_types, dict):
        raise ValueError('Invalid spike_types input.')

    spike_mask = {}
    for (s_label, s_list) in spike_types.items():
        s_label_mask = []
        for s_type in s_list:
            s_mask_list = []
            for (s_key, s_mask) in spike_types_mask.items():
                if s_key.startswith(s_type):
                    s_mask_list.append(s_mask)
            s_mask_list = np.logical_or.reduce(s_mask_list)
            s_label_mask.append(s_mask_list)
        spike_mask[s_label] = np.logical_or.reduce(s_label_mask)

    bins = np.linspace(0, spike_times[-1], 50)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    color_cycle = cycler('color', ['r', 'g', 'b', 'y', 'm', 'c'])
    plt.rcParams["axes.prop_cycle"] = color_cycle

    for (s_key, s_mask) in spike_mask.items():
        if np.any(s_mask):
            ax.hist(spike_times[s_mask], bins, label=s_key)

    plt.legend()
    if show:
        plt.show()
    return ax.get_figure()


def plot_spikes_raster(spikes, ax=None, show=True):
    """Plot the aggregate spiking activity according to cell type.

    Parameters
    ----------
    spikes : instance of Spikes
        The spikes object from net.spikes
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
    spike_times = np.array(sum(spikes._times, []))
    spike_types = np.array(sum(spikes._types, []))
    cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
    spike_times_cell = [spike_times[spike_types == cell_type]
                        for cell_type in cell_types]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.eventplot(spike_times_cell, colors=['r', 'b', 'g', 'w'])
    ax.legend(cell_types, ncol=2)
    ax.set_facecolor('k')
    ax.set_xlabel('Time (ms)')
    ax.get_yaxis().set_visible(False)
    ax.set_ylim((-1, 4.5))
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

    for cell_type in net.pos_dict:
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
