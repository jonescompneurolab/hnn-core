"""Visualization functions."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np


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
            String Input: 'common', 'evdist', 'evprox', 'extgauss', 'extpois'
        List of valid spike types will plot each type individually.
            List Input: ['common', 'evdist']
        Dictionary of valid lists will plot list elements as a group.
            Dictionary Input: {'Evoked': ['evdist', 'evprox']}
        If None, all spike types are plotted individually.
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    spike_times = np.array(sum(spikes._times, []))
    spike_types_array = np.array(sum(spikes._types, []))
    spike_types_unique = np.unique(spike_types_array)
    
    labels = {
        'common': 'Common', 'evdist': 'Distal',
        'evprox': 'Proximal', 'extgauss': 'External Gaussian',
        'extpois': 'External Poisson'}
    if type(spike_types) is str:
        spike_types = {labels[spike_types]: spike_types_unique[np.in1d(spike_types_unqiue, spike_types)] }
    elif type(spike_types) is list:
        spike_types = {labels[spike_types]: spike_types_unique[np.in1d(spike_types_unqiue, spike_types)] } #incomplete
    elif spike_types is None:
        spike_types = {s_label: [s_type] for s_type, s_label in labels.items()}

    #**Check how to grab this from spikes**
    bins = np.linspace(0, spike_times[0], 50)

    if ax is None:
        fig, ax = plt.subplots(1, 1)


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
