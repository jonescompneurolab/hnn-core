"""Visualization functions."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

import numpy as np
from itertools import cycle
import colorsys
import warnings
from .externals.mne import _validate_type


def _lighten_color(color, amount=0.5):
    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def _get_plot_data_trange(times, data, tmin=None, tmax=None):
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
            raise ValueError(
                "each decimation factor must be a positive int, "
                f"but {dec} is a {type(dec)}"
            )
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

    if show and get_backend() != "agg":
        (fig or plt).show(**kwargs)


def plot_laminar_lfp(
    times,
    data,
    contact_labels,
    tmin=None,
    tmax=None,
    ax=None,
    decim=None,
    color="cividis",
    voltage_offset=50,
    voltage_scalebar=200,
    show=True,
):
    """Plot laminar extracellular electrode array voltage time series.

    Parameters
    ----------
    times : array-like, shape (n_times,)
        Sampling times (in ms).
    data : Two-dimensional Numpy array
        The extracellular voltages as an (n_contacts, n_times) array.
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

    _validate_type(times, (list, np.ndarray), "times")
    _validate_type(data, (list, np.ndarray), "data")
    if isinstance(times, list):
        times = np.array(times)
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if len(times) != data.shape[1]:
        raise ValueError(
            f"length of times ({len(times)}) and data ({len(data)}) do not match"
        )

    n_contacts = data.shape[0]
    if color is not None:
        _validate_type(color, (str, tuple, list, np.ndarray, ListedColormap), "color")
        if isinstance(color, (tuple, list)):
            if (
                not np.all([isinstance(c, float) for c in color])
                or len(color) < 3
                or len(color) > 4
            ):
                raise ValueError(f"color must be length 3 or 4, got {color}")
        elif isinstance(color, np.ndarray):
            if color.shape[0] != n_contacts or (
                color.shape[1] < 3 or color.shape[1] > 4
            ):
                raise ValueError(f"color must be n_contacts x (3 or 4), got {color}")
        elif isinstance(color, ListedColormap):
            if color.N != n_contacts:
                raise ValueError(
                    f"ListedColormap has N={color.N}, but "
                    f"there are {n_contacts} contacts"
                )
        elif isinstance(color, str):
            color = plt.get_cmap(color, len(contact_labels))

    if ax is None:
        _, ax = plt.subplots(1, 1)

    n_offsets = data.shape[0]
    trace_offsets = np.zeros((n_offsets, 1))
    if voltage_offset is not None:
        trace_offsets = np.arange(n_offsets)[:, np.newaxis] * voltage_offset

    for contact_no, trace in enumerate(np.atleast_2d(data)):
        plot_data = trace
        plot_times = times

        if decim is not None:
            plot_data, plot_times = _decimate_plot_data(decim, plot_data, plot_times)

        if isinstance(color, np.ndarray):
            col = color[contact_no]
        elif isinstance(color, ListedColormap):
            col = color(contact_no)
        else:
            col = color
        ax.plot(
            plot_times,
            plot_data + trace_offsets[contact_no],
            label=f"C{contact_no}",
            color=col,
        )

        # To be removed after deprecation cycle
        if tmin is not None or tmax is not None:
            ax.set_xlim(left=tmin, right=tmax)
            warnings.warn(
                "tmin and tmax are deprecated and will be "
                "removed in future releases of hnn-core. Please"
                "use matplotlib plt.xlim to set tmin and tmax.",
                DeprecationWarning,
            )

        else:
            ax.set_xlim(left=times[0], right=times[-1])
    if voltage_offset is not None:
        ax.set_ylim(-voltage_offset, n_offsets * voltage_offset)
        ylabel = "Individual contact traces"
        if len(contact_labels) != n_offsets:
            raise ValueError(
                f"contact_labels is length {len(contact_labels)},"
                f" but {n_offsets} contacts to be plotted"
            )
        else:
            trace_ticks = np.arange(
                0, len(contact_labels) * voltage_offset, voltage_offset
            )
            ax.set_yticks(trace_ticks)
            ax.set_yticklabels(contact_labels)

        if voltage_scalebar is None:
            voltage_scalebar = voltage_offset

    if voltage_scalebar is not None:
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        scalebar = AnchoredSizeBar(
            ax.transData,
            1,
            f"{voltage_scalebar:.0f} " + r"$\mu V$",
            "upper left",
            size_vertical=voltage_scalebar,
            pad=0.1,
            color="black",
            label_top=False,
            frameon=False,
        )
        ax.add_artist(scalebar)
    else:
        ylabel = r"Electric potential ($\mu V$)"
        ax.ticklabel_format(axis="both", scilimits=(-2, 3))

    ax.set_ylabel(ylabel, multialignment="center")
    ax.set_xlabel("Time (ms)")

    plt_show(show)
    return ax.get_figure()


def plot_dipole(
    dpl,
    tmin=None,
    tmax=None,
    ax=None,
    layer="agg",
    decim=None,
    color="k",
    label="average",
    average=False,
    show=True,
):
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
        _, ax = plt.subplots(
            len(layers), 1, constrained_layout=True, sharex=True, sharey=True
        )
    axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

    if isinstance(dpl, Dipole):
        dpl = [dpl]
    for this_dpl in dpl:
        _validate_type(this_dpl, Dipole, "dpl", "Dipole, list of Dipole")

    if average:
        dpl.append(average_dipoles(dpl))

    scale_applied = dpl[0].scale_applied

    assert len(layers) == len(axes), "ax and layer should have the same size"

    for layer, ax in zip(layers, axes):
        for idx, dpl_trial in enumerate(dpl):
            if dpl_trial.scale_applied != scale_applied:
                raise RuntimeError("All dipoles must be scaled equally!")

            if layer in dpl_trial.data.keys():
                # extract scaled data and times
                data = dpl_trial.data[layer]
                times = dpl_trial.times

                if decim is not None:
                    data, times = _decimate_plot_data(decim, data, times)
                if idx == len(dpl) - 1 and average:
                    # the average dpl
                    ax.plot(times, data, color=color, label=label, lw=1.5)
                else:
                    alpha = 0.5 if average else 1.0
                    ax.plot(
                        times,
                        data,
                        color=_lighten_color(color, 0.5),
                        alpha=alpha,
                        lw=1.0,
                    )

            # To be removed after deprecation cycle
            if tmin is not None or tmax is not None:
                if tmin is not None or tmax is not None:
                    warnings.warn(
                        "tmin and tmax are deprecated and will be "
                        "removed in future releases of hnn-core. "
                        "Please use matplotlib plt.xlim to set tmin"
                        " and tmax.",
                        DeprecationWarning,
                    )
                ax.set_xlim(left=tmin, right=tmax)
            else:
                ax.set_xlim(left=0, right=times[-1])
        if average:
            ax.legend()

        ax.ticklabel_format(axis="both", scilimits=(-2, 3))
        ax.set_xlabel("Time (ms)")
        if scale_applied == 1:
            ylabel = "Dipole moment (nAm)"
        else:
            ylabel = "Dipole moment\n(nAm " + r"$\times$ {:.0f})".format(scale_applied)
        ax.set_ylabel(ylabel, multialignment="center")
        if layer == "agg":
            title_str = "Aggregate (L2/3 + L5)"
        elif layer == "L2":
            title_str = "L2/3"
        else:
            title_str = layer
        ax.set_title(title_str)

    plt_show(show)
    return axes[0].get_figure()


def plot_spikes_hist(
    cell_response,
    trial_idx=None,
    ax=None,
    spike_types=None,
    color=None,
    invert_spike_types=None,
    show=True,
    **kwargs_hist,
):
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
    invert_spike_types: string | list | None
        String input of a valid spike type to be mirrored about the y axis

        | Ex: ``'evdist'``, ``'evprox'``, ...

        List of valid spike types to be mirrored about the y axis

        | Ex: ``['evdist', 'evprox']``

        If None, all input spike types are plotted on the same y axis
    color : str | list of str | dict | None
        Input defining colors of plotted histograms. If str, all
        histograms plotted with same color. If list of str provided,
        histograms for each spike type will be plotted by cycling
        through colors in the list.

        If dict, colors must be specified for all spike_types as a key.
        If a group of spike types is defined by the `spike_types`
        parameter (see dictionary example for `spike_types`),
        the name of this group must be used to specify the colors.

        | Ex: ``{'evdist': 'g', 'evprox': 'r'}``, ``{'Tonic': 'b'}``

        If None, default color cycle used.
    show : bool
        If True, show the figure.
    **kwargs_hist : dict
        Additional keyword arguments to pass to ax.hist.

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
    _validate_type(trial_idx, list, "trial_idx", "int, list of int")

    # Extract desired trials
    if len(cell_response._spike_times[0]) > 0:
        spike_times = np.concatenate(
            np.array(cell_response._spike_times, dtype=object)[trial_idx]
        )
        spike_types_data = np.concatenate(
            np.array(cell_response._spike_types, dtype=object)[trial_idx]
        )
    else:
        spike_times = np.array([])
        spike_types_data = np.array([])

    unique_types = np.unique(spike_types_data)
    spike_types_mask = {
        s_type: np.isin(spike_types_data, s_type) for s_type in unique_types
    }
    cell_types = ["L5_pyramidal", "L5_basket", "L2_pyramidal", "L2_basket"]
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
                raise TypeError(
                    f"spike_types[{spike_label}] must be a list. "
                    f"Got "
                    f"{type(spike_types[spike_label]).__name__}."
                )

    if not isinstance(spike_types, dict):
        raise TypeError("spike_types should be str, list, dict, or None")

    spike_labels = dict()
    for spike_label, spike_type_list in spike_types.items():
        for spike_type in spike_type_list:
            n_found = 0
            for unique_type in unique_types:
                if unique_type.startswith(spike_type):
                    if unique_type in spike_labels:
                        raise ValueError(
                            f"Elements of spike_types must map to"
                            f" mutually exclusive input types."
                            f" {unique_type} is found more than"
                            f" once."
                        )
                    spike_labels[unique_type] = spike_label
                    n_found += 1
            if n_found == 0:
                raise ValueError(f"No input types found for {spike_type}")

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    _validate_type(color, (str, list, dict, None), "color", "str, list of str, or dict")

    if color is None:
        color_cycle = cycle(["r", "g", "b", "y", "m", "c"])
    elif isinstance(color, str):
        color_cycle = cycle([color])
    elif isinstance(color, list):
        color_cycle = cycle(color)

    if len(cell_response.times) > 0:
        bins = np.linspace(0, cell_response.times[-1], 50)
    else:
        bins = np.linspace(0, spike_times[-1], 50)

    # Create dictionary to aggregate spike times that have the same spike_label
    spike_type_times = {
        spike_label: list() for spike_label in np.unique(list(spike_labels.values()))
    }
    spike_color = dict()  # Store colors specified for each spike_label
    for spike_type, spike_label in spike_labels.items():
        if spike_label not in spike_color:
            if isinstance(color, dict):
                if spike_label not in color:
                    raise ValueError(
                        f"'{spike_label}' must be defined in color dictionary"
                    )
                _validate_type(
                    color[spike_label], str, "Dictionary values of color", "str"
                )
                spike_color[spike_label] = color[spike_label]
            else:
                spike_color[spike_label] = next(color_cycle)
        spike_type_times[spike_label].extend(spike_times[spike_types_mask[spike_type]])

    if invert_spike_types is None:
        invert_spike_types = list()
    else:
        if not isinstance(invert_spike_types, (str, list)):
            raise TypeError(
                "'invert_spike_types' must be a string or a list of strings"
            )
        if isinstance(invert_spike_types, str):
            invert_spike_types = [invert_spike_types]

        # Check that spike types to invert are correctly specified
        unique_inputs = set(spike_labels.values())
        unique_invert_inputs = set(invert_spike_types)
        check_intersection = unique_invert_inputs.intersection(unique_inputs)
        if not check_intersection == unique_invert_inputs:
            raise ValueError(
                "Elements of 'invert_spike_types' mustmap to valid input types"
            )

    # Initialize secondary axis
    ax1 = None

    # Plot aggregated spike_times
    for spike_label, plot_data in spike_type_times.items():
        hist_color = spike_color[spike_label]

        # Plot on the primary y-axis
        if spike_label not in invert_spike_types:
            ax.hist(plot_data, bins, label=spike_label, color=hist_color, **kwargs_hist)
        # Plot on secondary y-axis
        else:
            if ax1 is None:
                ax1 = ax.twinx()
            ax1.hist(
                plot_data, bins, label=spike_label, color=hist_color, **kwargs_hist
            )
            # Need to add label for easy removal later

    # Set the y-limits based on the maximum across both axes
    if ax1 is not None:
        ax_ylim = ax.get_ylim()[1]
        ax1_ylim = ax1.get_ylim()[1]

        y_max = max(ax_ylim, ax1_ylim)
        ax.set_ylim(0, y_max)
        ax1.set_ylim(0, y_max)
        ax1.invert_yaxis()
        ax1.set_label("Inverted spike histogram")

    if len(cell_response.times) > 0:
        ax.set_xlim(left=0, right=cell_response.times[-1])
    else:
        ax.set_xlim(left=0)

    ax.set_ylabel("Counts")
    ax.set_label("Spike histogram")

    if ax1 is not None:
        # Combine legends
        handles, labels = ax.get_legend_handles_labels()
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles.extend(handles1)
        labels.extend(labels1)

        ax1.legend(handles, labels, loc="upper left")
    else:
        ax.legend()

    plt_show(show)
    return ax.get_figure()


def plot_spikes_raster(
    cell_response,
    trial_idx=None,
    ax=None,
    show=True,
    cell_types=None,
    colors=None,
    show_legend=True,
    marker_size=1.0,
    dpl=None,
    overlay_dipoles=False,
):
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
    cell_types : list of str
        List of cell types to plot
    colors : list of str | None
        Optional custom colors to plot. Default will use the color cycler.
    show_legend : bool
        If True, show the legend with colors for cell types
    marker_size : float
        Optional marker size to use when plotting spikes. Uses
        "linelengths" argument of ax.eventplot, which accepts positive
        numeric values only
    dpl : instance of Dipole | list
        The Dipole object containing layer-specific dipole data
        to overlay on the raster plot
    overlay_dipoles : bool
        If True, overlay the layer-specific dipole data on the
        raster plot

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure object.
    """

    import matplotlib.pyplot as plt
    from .dipole import Dipole, average_dipoles

    n_trials = len(cell_response.spike_times)
    if trial_idx is None:
        trial_idx = list(range(n_trials))

    # Get spike types from cell response
    unique_spike_types = cell_response.cell_types

    # validate trial argument
    if isinstance(trial_idx, int):
        trial_idx = [trial_idx]
    _validate_type(trial_idx, list, "trial_idx", "int, list of int")

    # validate cell types
    if cell_types:
        _validate_type(cell_types, list, "cell_types", "list of str")
        if not set(cell_types).issubset(set(unique_spike_types)):
            raise ValueError(
                "Invalid cell types provided. "
                f"Must be of set {unique_spike_types}. "
                f"Got {cell_types}"
            )
    else:
        # Use default cell types
        cell_types = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]

    # Set default colors
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][
        : len(cell_types)
    ]
    cell_colors = {cell: color for cell, color in zip(cell_types, default_colors)}

    # validate colors argument
    _validate_type(colors, (list, dict, None), "color", "list of str, or dict")
    if colors:
        if isinstance(colors, list):
            if len(colors) != len(cell_types):
                raise ValueError(
                    f"Number of colors must be equal to number of "
                    f"cell types. {len(colors)} colors provided "
                    f"for {len(cell_types)} cell types."
                )
            cell_colors = {cell: color for cell, color in zip(cell_types, colors)}

        if isinstance(colors, dict):
            # Check valid cell types
            if not set(colors.keys()).issubset(set(unique_spike_types)):
                raise ValueError(
                    "Invalid cell types provided. "
                    f"Must be of set {unique_spike_types}. "
                    f"Got {colors.keys()}"
                )
            cell_colors.update(colors)

    # validate show_legend argument
    _validate_type(show_legend, bool, "show_legend", "bool")

    # validate marker_size
    _validate_type(marker_size, (float, int), "marker_size", "float or int")

    # if marker_size is <= 0, set it to the default value of 1.0
    if marker_size <= 0:
        marker_size = 1.0

    # Extract desired trials
    spike_times = np.concatenate(
        np.array(cell_response._spike_times, dtype=object)[trial_idx]
    )
    spike_types = np.concatenate(
        np.array(cell_response._spike_types, dtype=object)[trial_idx]
    )
    spike_gids = np.concatenate(
        np.array(cell_response._spike_gids, dtype=object)[trial_idx]
    )

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    events = []
    for cell_type, color in cell_colors.items():
        cell_type_gids = np.unique(spike_gids[spike_types == cell_type])
        cell_type_times, cell_type_ypos = [], []

        for gid in cell_type_gids:
            gid_time = spike_times[spike_gids == gid]
            cell_type_times.append(gid_time)
            cell_type_ypos.append(-gid)

        if cell_type_times:
            events.append(
                ax.eventplot(
                    cell_type_times,
                    lineoffsets=cell_type_ypos,
                    color=color,
                    label=cell_type,
                    linelengths=marker_size,
                )
            )
        else:
            # Blank plot for no spiking
            events.append(
                ax.eventplot(
                    [-1], lineoffsets=[-1], color=color, label=cell_type, linelengths=1
                )
            )

    # Overlay dipoles on raster plot
    if overlay_dipoles:
        # Confirm that the dpl is not None
        if dpl is None:
            raise ValueError("Dipole object must be provided to overlay dipoles")
        # In the case that a single Dipole object is provided, cast to a list
        if isinstance(dpl, Dipole):
            dpl = [dpl]
        # Validate that all list elements are Dipole objects
        for trial in dpl:
            _validate_type(trial, Dipole, "dpl", "Dipole, list of Dipole")

        # Get (average) layer-sepcific dipole data
        avgs = []
        avgs.append(average_dipoles(dpl))
        l2_dipole = avgs[0].data["L2"]
        l5_dipole = avgs[0].data["L5"]
        dipole_times = dpl[0].times

        # Scale dipole to fit the spike raster plot
        raster_yrange = ax.get_yticks()
        raster_min = min(raster_yrange)
        raster_midpoint = round((raster_min / 2), 0)
        raster_quarterpoint = round((raster_min / 4), 0)

        # Scale down by .95 until the dipoles fit within the appropriate area
        while (
            max(max(l5_dipole), max(l2_dipole)) - min(min(l5_dipole), min(l2_dipole))
        ) > abs(raster_midpoint):
            l5_dipole = l5_dipole * 0.95
            l2_dipole = l2_dipole * 0.95

        # Shift the dipole positions to overlay the correct cell types
        l2_dipole = l2_dipole - abs(raster_midpoint) + abs(raster_quarterpoint)
        l5_dipole = l5_dipole - abs(raster_midpoint) - abs(raster_quarterpoint)

        # Draw the dipole plots
        (l2_line,) = ax.plot(
            dipole_times,
            l2_dipole,
            color="grey",
            linewidth=1,
            linestyle="--",
            label="L2 Dipole",
        )
        (l5_line,) = ax.plot(
            dipole_times,
            l5_dipole,
            color="grey",
            linewidth=1,
            linestyle="-",
            label="L5 Dipole",
        )

    # set legend
    handles = [e[0] for e in events]

    # Add labels for the spike events
    for handle in handles:
        handle.set_label(handle.get_label() + " Spikes")

    if overlay_dipoles:
        handles.append(l2_line)
        handles.append(l5_line)

    spike_legend = ax.legend(
        handles=handles,
        loc="lower left",
        framealpha=0.25,
    )
    if not show_legend:
        ax.get_legend().remove()
    else:
        ax.add_artist(spike_legend)

    # set axis labels
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cell ID")

    # hide y-axis ticks and tick labels
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)

    # add title
    if overlay_dipoles:
        ax.set_title("Raster Plot with Layer-Specific Dipole Overlays")
    else:
        ax.set_title("Raster Plot")

    if len(cell_response.times) > 0:
        ax.set_xlim(left=0, right=cell_response.times[-1])
    else:
        ax.set_xlim(left=0)
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
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    elif not isinstance(ax, Axes3D):
        raise TypeError(
            f"Expected 'ax' to be an instance of Axes3D, but got {type(ax).__name__}"
        )

    colors = {
        "L5_pyramidal": "b",
        "L2_pyramidal": "c",
        "L5_basket": "r",
        "L2_basket": "m",
    }
    markers = {
        "L5_pyramidal": "^",
        "L2_pyramidal": "^",
        "L5_basket": "x",
        "L2_basket": "x",
    }

    for cell_type in net.cell_types:
        x = [pos[0] for pos in net.pos_dict[cell_type]]
        y = [pos[1] for pos in net.pos_dict[cell_type]]
        z = [pos[2] for pos in net.pos_dict[cell_type]]
        if cell_type in colors:
            color = colors[cell_type]
            marker = markers[cell_type]
            ax.scatter(x, y, z, c=color, s=50, marker=marker, label=cell_type)

    if net.rec_arrays:
        cols = plt.get_cmap("inferno", len(net.rec_arrays) + 2)
        for ii, (arr_name, arr) in enumerate(net.rec_arrays.items()):
            x = [p[0] for p in arr.positions]
            y = [p[1] for p in arr.positions]
            z = [p[2] for p in arr.positions]
            ax.scatter(x, y, z, color=cols(ii + 1), s=25, marker="o", label=arr_name)

    plt.legend(bbox_to_anchor=(-0.15, 1.025), loc="upper left")

    plt_show(show)
    return ax.get_figure()


def plot_tfr_morlet(
    dpl,
    freqs,
    *,
    n_cycles=7.0,
    tmin=None,
    tmax=None,
    layer="agg",
    decim=None,
    padding="zeros",
    ax=None,
    colormap="inferno",
    colorbar=True,
    colorbar_inside=False,
    show=True,
):
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
            raise RuntimeError("All dipoles must be scaled equally!")
        if dpl_trial.sfreq != sfreq:
            raise RuntimeError("All dipoles must be sampled equally!")

        data, times = _get_plot_data_trange(
            dpl_trial.times, dpl_trial.data[layer], tmin, tmax
        )

        sfreq = dpl_trial.sfreq
        if decim is not None:
            data, times, sfreq = _decimate_plot_data(decim, data, times, sfreq=sfreq)

        if padding is not None:
            if not isinstance(padding, str):
                raise ValueError("padding must be a string (or None)")
            if padding == "zeros":
                data = np.r_[
                    np.zeros((len(data) - 1,)), data.ravel(), np.zeros((len(data) - 1,))
                ]
            elif padding == "mirror":
                data = np.r_[data[-1:0:-1], data, data[-2::-1]]

        # MNE expects an array of shape (n_trials, n_channels, n_times)
        data = data[None, None, :]
        power = tfr_array_morlet(
            data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output="power"
        )

        if padding is not None:
            # get the middle portion after padding
            power = power[:, :, :, times.shape[0] - 1 : 2 * times.shape[0] - 1]
        trial_power.append(power)

    power = np.mean(trial_power, axis=0)
    im = ax.pcolormesh(times, freqs, power[0, 0, ...], cmap=colormap, shading="auto")

    if freqs[0] > freqs[-1]:
        freqs = freqs[::-1]
        ax.invert_yaxis()

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        # default colorbar
        if colorbar_inside is False:
            cbar = fig.colorbar(im, ax=ax, format=xfmt, shrink=0.8, pad=0)
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.set_ylabel(
                r"Power ([nAm $\times$ {:.0f}]$^2$)".format(scale_applied),
                rotation=-90,
                va="bottom",
            )
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
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.offsetText.set_fontsize(cbar_fontsize)

            cbar.ax.set_ylabel(
                r"Power ([nAm $\times$ {:.0f}]$^2$)".format(scale_applied),
                rotation=-90,
                va="bottom",
                fontsize=cbar_fontsize,
                color=cbar_color,
            )
            cbar.ax.tick_params(
                direction="in",
                labelsize=cbar_fontsize,
                labelcolor=cbar_color,
                colors=cbar_color,
            )
            plt.setp(cbar.ax.spines.values(), color=cbar_color)
            setattr(fig, f"_cbar-ax-{id(ax)}", cbar)

    plt_show(show)
    return ax.get_figure()


def plot_psd(
    dpl,
    *,
    fmin=0,
    fmax=None,
    tmin=None,
    tmax=None,
    layer="agg",
    color=None,
    label=None,
    ax=None,
    show=True,
):
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
            raise RuntimeError("All dipoles must be scaled equally!")
        if dpl_trial.sfreq != sfreq:
            raise RuntimeError("All dipoles must be sampled equally!")

        data, _ = _get_plot_data_trange(
            dpl_trial.times, dpl_trial.data[layer], tmin, tmax
        )

        freqs, Pxx = periodogram(data, sfreq, window="hamming", nfft=len(data))
        trial_power.append(Pxx)

    ax.plot(freqs, np.mean(np.array(Pxx, ndmin=2), axis=0), color=color, label=label)
    if label:
        ax.legend()
    if fmax is not None:
        ax.set_xlim((fmin, fmax))
    ax.ticklabel_format(axis="both", scilimits=(-2, 3))
    ax.set_xlabel("Frequency (Hz)")
    if scale_applied == 1:
        ylabel = "Power spectral density\n(nAm" + r"$^2 \ Hz^{-1}$)"
    else:
        ylabel = (
            "Power spectral density\n"
            + r"([nAm$\times$ {:.0f}]".format(scale_applied)
            + r"$^2 \ Hz^{-1}$)"
        )
    ax.set_ylabel(ylabel, multialignment="center")

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


def plot_cell_morphology(
    cell,
    ax,
    color=None,
    pos=(0, 0, 0),
    xlim=(-250, 150),
    ylim=(-100, 100),
    zlim=(-100, 1200),
    show=True,
):
    """Plot the cell morphology.

    Parameters
    ----------
    cell : instance of Cell
        The cell object
    ax : instance of Axes3D
        Matplotlib 3D axis
    show : bool
        If True, show the plot
    color : str | dict | None
        Color of cell. If str, entire cell plotted with
        color indicated by str. If dict, colors of individual sections
        can be specified. Must have a key for every section in cell as
        defined in the `Cell.sections` attribute.

        | Ex: ``{'apical_trunk': 'r', 'soma': 'b', ...}``
    pos : tuple of int or float | None
        Position of cell soma. Must be a tuple of 3 elements for the
        (x, y, z) position of the soma in 3D space. Default: (0, 0, 0)
    xlim : tuple of int | tuple of float
        x limits of plot window. Default (-250, 150)
    ylim : tuple of int | tuple of float
        y limits of plot window. Default (-100, 100)
    zlim : tuple of int | tuple of float
        z limits of plot window. Default (-100, 1200)
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
        ax = plt.axes(projection="3d")

    _validate_type(color, (str, dict, None), "color")

    if color is None:
        section_colors = {section: "b" for section in cell.sections.keys()}
    if isinstance(color, str):
        section_colors = {section: color for section in cell.sections.keys()}
    if isinstance(color, dict):
        section_colors = color

    _validate_type(pos, tuple, "pos")
    if isinstance(pos, tuple):
        if len(pos) != 3:
            raise ValueError("pos must be a tuple of 3 elements")
        for pos_idx in pos:
            _validate_type(pos_idx, (float, int), "pos[idx]")

    # Cell is in XZ plane
    ax.set_xlim((pos[0] + xlim[0], pos[0] + xlim[1]))
    ax.set_zlim((pos[1] + zlim[0], pos[1] + zlim[1]))
    ax.set_ylim((pos[2] + ylim[0], pos[2] + ylim[1]))

    for sec_name, section in cell.sections.items():
        linewidth = _linewidth_from_data_units(ax, section.diam)
        end_pts = section.end_pts
        dx = pos[0] - cell.sections["soma"].end_pts[0][0]
        dy = pos[1] - cell.sections["soma"].end_pts[0][1]
        dz = pos[2] - cell.sections["soma"].end_pts[0][2]
        xs, ys, zs = list(), list(), list()
        for pt in end_pts:
            xs.append(pt[0] + dx)
            ys.append(pt[1] + dz)
            zs.append(pt[2] + dy)
        ax.plot(xs, ys, zs, "-", linewidth=linewidth, color=section_colors[sec_name])
    ax.view_init(0, -90)
    ax.axis("off")

    plt.tight_layout()
    plt_show(show)
    return ax


def plot_connectivity_matrix(
    net, conn_idx, ax=None, show_weight=True, colorbar=True, colormap="Greys", show=True
):
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

    _validate_type(net, Network, "net", "Network")
    _validate_type(conn_idx, int, "conn_idx", "int")
    _validate_type(show_weight, bool, "show_weight", "bool")
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Load objects for distance calculation
    conn = net.connectivity[conn_idx]
    nc_dict = conn["nc_dict"]
    src_type = conn["src_type"]
    target_type = conn["target_type"]
    src_type_pos = net.pos_dict[src_type]
    target_type_pos = net.pos_dict[target_type]

    src_range = np.array(net.gid_ranges[conn["src_type"]])
    target_range = np.array(net.gid_ranges[conn["target_type"]])
    connectivity_matrix = np.zeros((len(src_range), len(target_range)))

    for src_gid, target_src_pair in conn["gid_pairs"].items():
        src_idx = np.where(src_range == src_gid)[0][0]
        target_indeces = np.where(np.isin(target_range, target_src_pair))[0]
        for target_idx in target_indeces:
            src_pos = src_type_pos[src_idx]
            target_pos = target_type_pos[target_idx]

            # Identical calculation used in Cell.par_connect_from_src()
            if show_weight:
                weight, _ = _get_gaussian_connection(
                    src_pos, target_pos, nc_dict, inplane_distance=net._inplane_distance
                )
            else:
                weight = 1.0

            connectivity_matrix[src_idx, target_idx] = weight

    im = ax.imshow(connectivity_matrix, cmap=colormap, interpolation="none")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        cbar = fig.colorbar(im, ax=ax, format=xfmt)
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.set_ylabel("Weight", rotation=-90, va="bottom")

    ax.set_xlabel(
        f"{conn['target_type']} target gids ({target_range[0]}-{target_range[-1]})"
    )
    ax.set_xticklabels(list())
    ax.set_ylabel(f"{conn['src_type']} source gids ({src_range[0]}-{src_range[-1]})")
    ax.set_yticklabels(list())
    ax.set_title(
        f"{conn['src_type']} -> {conn['target_type']} "
        f"({conn['loc']}, {conn['receptor']})"
    )

    plt.tight_layout()
    plt_show(show)
    return ax.get_figure()


def plot_drive_strength(
    net,
    show_weight=True,
    ax=None,
    colorbar=True,
    color_scale="linear",
    normalize=True,
    show=True,
):
    """Plot the relative strength of drives to cell types.

    Parameters
    ----------
    net : Network
        Instance of a Network object.
    show_weight : bool, default=True
        If True, visualize connectivity weights as gradient.
        If False, all weights set to constant value.
    ax : matplotlib.Axes, optional
        Matplotlib axes. If None, create a new figure.
    colorbar : bool, default=True
        If True (default), adjust figure to include colorbar.
    color_scale : {'linear', 'log'}, default='linear'
        Color scaling for drive strength.
    normalize : bool, default=True
        If True, normalize the strength values to be between 0 and 1.
    show : bool, default=True
        If True, show the plot immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .network import Network
    from .cell import _get_gaussian_connection

    _validate_type(net, Network, "net", "Network")
    _validate_type(show_weight, bool, "show_weight", "bool")

    # Get drive and cell type information
    drive_names = list(net.external_drives.keys())
    cell_types = list(net.cell_types.keys())

    strength_matrix = np.zeros((len(drive_names), len(cell_types)))

    for conn in net.connectivity:
        src_type = conn["src_type"]
        target_type = conn["target_type"]
        if src_type in drive_names and target_type in cell_types:
            nc_dict = conn["nc_dict"]
            drive_idx = drive_names.index(src_type)
            cell_idx = cell_types.index(target_type)
            src_type_pos = net.pos_dict[src_type]
            target_type_pos = net.pos_dict[target_type]

            src_range = np.array(net.gid_ranges[src_type])
            target_range = np.array(net.gid_ranges[target_type])

            total_weight = 0
            for src_gid, target_src_pair in conn["gid_pairs"].items():
                src_idx = np.where(src_range == src_gid)[0][0]
                target_indeces = np.where(np.isin(target_range, target_src_pair))[0]
                for target_idx in target_indeces:
                    src_pos = src_type_pos[src_idx]
                    target_pos = target_type_pos[target_idx]

                    if show_weight:
                        weight = _get_gaussian_connection(
                            src_pos,
                            target_pos,
                            nc_dict,
                            inplane_distance=net._inplane_distance,
                        )
                    else:
                        weight = 1.0

                    total_weight += sum(weight)
                strength_matrix[drive_idx, cell_idx] = total_weight

    if normalize:
        min_val = np.min(strength_matrix)
        max_val = np.max(strength_matrix)
        if max_val > min_val:
            strength_matrix = (strength_matrix - min_val) / (max_val - min_val)

    if color_scale == "log":
        strength_matrix = np.log1p(strength_matrix)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    cax = ax.imshow(
        strength_matrix, cmap="viridis", aspect="auto", interpolation="nearest"
    )

    if colorbar:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.ax.yaxis.set_ticks_position("right")

    ax.set_xticks(np.arange(len(cell_types)))
    ax.set_yticks(np.arange(len(drive_names)))
    ax.set_xticklabels(cell_types)
    ax.set_yticklabels(drive_names)
    ax.set_xlabel("Cell Types")
    ax.set_ylabel("Drive Names")
    ax.set_title("External Drive Strengths")

    # For addition of units information
    unit_text = (
        "Units: µS (microsiemens)"
        if not normalize
        else "Units: Relative Strength (0-1)"
    )
    ax.text(
        0.98,
        0.02,
        unit_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(drive_names)):
        for j in range(len(cell_types)):
            value = strength_matrix[i, j]
            color = "black" if value > strength_matrix.max() * 0.7 else "white"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                weight="bold",
            )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _update_target_plot(
    ax,
    conn,
    src_gid,
    src_type_pos,
    target_type_pos,
    src_range,
    target_range,
    nc_dict,
    colormap,
    inplane_distance,
):
    from .cell import _get_gaussian_connection

    # Extract indices to get position in network
    # Index in gid range aligns with net.pos_dict
    target_src_pair = conn["gid_pairs"][src_gid]
    target_indeces = np.where(np.isin(target_range, target_src_pair))[0]

    src_idx = np.where(src_range == src_gid)[0][0]
    src_pos = src_type_pos[src_idx]

    # Aggregate positions and weight of each connected target
    weights, target_x_pos, target_y_pos = list(), list(), list()
    for target_idx in target_indeces:
        target_pos = target_type_pos[target_idx]
        target_x_pos.append(target_pos[0])
        target_y_pos.append(target_pos[1])
        weight, _ = _get_gaussian_connection(
            src_pos, target_pos, nc_dict, inplane_distance
        )
        weights.append(weight)

    ax.clear()
    im = ax.scatter(target_x_pos, target_y_pos, c=weights, s=50, cmap=colormap)
    x_pos = target_type_pos[:, 0]
    y_pos = target_type_pos[:, 1]
    ax.scatter(x_pos, y_pos, color="k", marker="x", zorder=-1, s=20)
    ax.scatter(src_pos[0], src_pos[1], marker="s", color="red", s=150)
    ax.set_ylabel("Y Position")
    ax.set_xlabel("X Position")
    return im


def plot_cell_connectivity(
    net, conn_idx, src_gid=None, axes=None, colorbar=True, colormap="viridis", show=True
):
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

    _validate_type(net, Network, "net", "Network")
    _validate_type(conn_idx, int, "conn_idx", "int")

    # Load objects for distance calculation
    conn = net.connectivity[conn_idx]
    nc_dict = conn["nc_dict"]
    src_type = conn["src_type"]
    target_type = conn["target_type"]
    src_type_pos = np.array(net.pos_dict[src_type])
    target_type_pos = np.array(net.pos_dict[target_type])
    src_range = np.array(net.gid_ranges[conn["src_type"]])

    valid_src_gids = list(net.connectivity[conn_idx]["gid_pairs"].keys())
    src_pos_valid = src_type_pos[np.isin(src_range, valid_src_gids)]

    if src_gid is None:
        src_gid = valid_src_gids[0]
    _validate_type(src_gid, int, "src_gid", "int")

    if src_gid not in valid_src_gids:
        raise ValueError(
            f"src_gid {src_gid} not a valid cell ID for this "
            f"connection. Please select one of {valid_src_gids}"
        )

    target_range = np.array(net.gid_ranges[conn["target_type"]])

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

    im = _update_target_plot(
        ax,
        conn,
        src_gid,
        src_type_pos,
        target_type_pos,
        src_range,
        target_range,
        nc_dict,
        colormap,
        net._inplane_distance,
    )

    x_src = src_type_pos[:, 0]
    y_src = src_type_pos[:, 1]
    x_src_valid = src_pos_valid[:, 0]
    y_src_valid = src_pos_valid[:, 1]
    if src_type in net.cell_types:
        ax_src.scatter(x_src, y_src, marker="s", color="red", s=50, alpha=0.2)
        ax_src.scatter(x_src_valid, y_src_valid, marker="s", color="red", s=50)

    plt.suptitle(
        f"{conn['src_type']}-> {conn['target_type']}"
        f" ({conn['loc']}, {conn['receptor']})"
    )

    def _onclick(event):
        if event.inaxes in [ax] or event.inaxes is None:
            return

        dist = np.linalg.norm(
            src_type_pos[:, :2] - np.array([event.xdata, event.ydata]), axis=1
        )
        src_idx = np.argmin(dist)

        src_gid = src_range[src_idx]
        if src_gid not in valid_src_gids:
            return
        _update_target_plot(
            ax,
            conn,
            src_gid,
            src_type_pos,
            target_type_pos,
            src_range,
            target_range,
            nc_dict,
            colormap,
            net._inplane_distance,
        )

        fig.canvas.draw()

    if colorbar:
        fig = ax.get_figure()
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-2, 2))
        cbar = fig.colorbar(im, ax=ax, format=xfmt)
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.set_ylabel("Weight", rotation=-90, va="bottom")

    plt.tight_layout()

    fig.canvas.mpl_connect("button_press_event", _onclick)

    plt_show(show)
    return ax.get_figure()


def plot_laminar_csd(
    times,
    data,
    contact_labels,
    ax=None,
    colorbar=True,
    vmin=None,
    vmax=None,
    sink="b",
    interpolation="spline",
    show=True,
):
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
        If True (default), adjust figure to include colorbar.
    contact_labels : list
        Labels associated with the contacts to plot. Passed as-is to
        :func:`~matplotlib.axes.Axes.set_yticklabels`.
    vmin: float, optional
        lower bound of the color axis.
        Will be set automatically of None.
    vmax: float, optional
        upper bound of the color axis.
        Will be set automatically of None.
    sink : str
        If set to 'blue' or 'b', plots sinks in blue and sources in red,
        if set to 'red' or 'r', sinks plotted in red and sources blue.
    interpolation : str | None
        If 'spline', will smoothen the CSD using spline method,
        if None, no smoothing will be applied.

    show : bool
        If True, show the plot.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    if sink[0].lower() == "b":
        cmap = "jet"
    elif sink[0].lower() == "r":
        cmap = "jet_r"
    elif sink[0].lower() != "b" or sink[0].lower() != "r":
        raise RuntimeError(
            'Please use sink = "b" or sink = "r".'
            ' Only colormap "jet" is supported for CSD.'
        )

    if interpolation == "spline":
        # create interpolation function
        interp_data = RectBivariateSpline(times, contact_labels, data.T)
        # increase number of contacts
        new_depths = np.linspace(
            contact_labels[0],
            contact_labels[-1],
            contact_labels[-1] - contact_labels[0],
        )
        # interpolate
        data = interp_data(times, new_depths).T
    elif interpolation is None:
        data = data
        new_depths = contact_labels

    # if vmin and vmax are both None, set colormap such that green = zero
    if vmin is None and vmax is None:
        vmin = -np.max(np.abs(data))
        vmax = np.max(np.abs(data))

    im = ax.pcolormesh(
        times, new_depths, data, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("electrode depth")
    if colorbar:
        color_axis = ax.inset_axes([1.05, 0, 0.02, 1], transform=ax.transAxes)
        plt.colorbar(im, ax=ax, cax=color_axis).set_label(r"$CSD (uV/um^{2})$")

    plt.tight_layout()
    plt_show(show)

    return ax.get_figure()


class NetworkPlotter:
    """Helper class to visualize full morphology of HNN model.

    Parameters
    ----------
    net : Instance of Network object
        The Network object
    ax : instance of matplotlib Axes3D | None
        An axis object from matplotlib. If None,
        a new figure is created.
    vmin : int | float
        Lower limit of colormap for plotting voltage
        Default: -100 mV
    vmax : int | float
        Upper limit of colormap for plotting voltage
        Default: 50 mV
    bg_color : str
        Background color of ax. Default: 'black'
    colorbar : bool
        If True (default), adjust figure to include colorbar.
    voltage_colormap : str
        Colormap used for plotting voltages
        Default: 'viridis'
    elev : int | float
        Elevation 3D plot viewpoint, default: 10
    azim : int | float
        Azimuth of 3D plot view point, default: 20
    xlim : tuple of int | tuple of float
        x limits of plot window. Default (-200, 3100)
    ylim : tuple of int | tuple of float
        y limits of plot window. Default (-200, 3100)
    zlim : tuple of int | tuple of float
        z limits of plot window. Default (-300, 2200)
    trial_idx : int
        Index of simulation trial plotted. Default: 0
    time_idx : int
        Index of time point plotted. Default: 0
    """

    def __init__(
        self,
        net,
        ax=None,
        vmin=-100,
        vmax=50,
        bg_color="black",
        colorbar=True,
        voltage_colormap="viridis",
        elev=10,
        azim=-500,
        xlim=(-200, 3100),
        ylim=(-200, 3100),
        zlim=(-300, 2200),
        trial_idx=0,
        time_idx=0,
    ):
        from matplotlib import colormaps

        self._validate_parameters(
            vmin,
            vmax,
            bg_color,
            voltage_colormap,
            colorbar,
            elev,
            azim,
            xlim,
            ylim,
            zlim,
            trial_idx,
            time_idx,
        )

        # Set init arguments
        self.net = net
        self.ax = ax
        self._vmin = vmin
        self._vmax = vmax
        self._bg_color = bg_color
        self._colorbar = colorbar
        self._voltage_colormap = voltage_colormap
        self._colormaps = colormaps
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim
        self._elev = elev
        self._azim = azim
        self._trial_idx = trial_idx
        self._time_idx = time_idx

        # Check if Network object is simulated
        self.times, self._vsec_recorded = self._check_network_simulation()

        # Initialize plots and colormap
        self.fig = None
        self._colormap = colormaps[voltage_colormap]
        self.vsec_array = self._get_voltages()
        self.color_array = self._colormap(self.vsec_array)

        self._initialize_plots()
        if self._colorbar:
            self._update_colorbar()
        else:
            self._cbar = None

    def _validate_parameters(
        self,
        vmin,
        vmax,
        bg_color,
        voltage_colormap,
        colorbar,
        elev,
        azim,
        xlim,
        ylim,
        zlim,
        trial_idx,
        time_idx,
    ):
        _validate_type(vmin, (int, float), "vmin")
        _validate_type(vmax, (int, float), "vmax")
        _validate_type(bg_color, str, "bg_color")
        _validate_type(voltage_colormap, str, "voltage_colormap")
        _validate_type(colorbar, bool, "colorbar")
        _validate_type(xlim, tuple, "xlim")
        _validate_type(ylim, tuple, "ylim")
        _validate_type(zlim, tuple, "zlim")
        _validate_type(elev, (int, float), "elev")
        _validate_type(azim, (int, float), "azim")
        _validate_type(trial_idx, int, "trial_idx")
        _validate_type(time_idx, int, "time_idx")

    def _check_network_simulation(self):
        times = None
        vsec_recorded = False
        # Check if network simulated
        if self.net.cell_response is not None:
            times = self.net.cell_response.times

            # Check if voltage recorded
            if self.net._params["record_vsec"] == "all":
                vsec_recorded = True
        return times, vsec_recorded

    def _initialize_plots(self):
        import matplotlib.pyplot as plt

        # Create figure
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.set_facecolor(self._bg_color)

        self._init_network_plot()
        self._update_axes()

    def _get_voltages(self):
        vsec_list = list()
        for cell_type in self.net.cell_types:
            gid_range = self.net.gid_ranges[cell_type]
            for gid in gid_range:
                cell = self.net.cell_types[cell_type]
                for sec_name in cell.sections.keys():
                    if self._vsec_recorded is True:
                        vsec = np.array(
                            self.net.cell_response.vsec[self.trial_idx][gid][sec_name]
                        )
                        vsec_list.append(vsec)
                    else:  # Populate with zeros if no voltage recording
                        vsec_list.append([0.0])

        vsec_array = np.vstack(vsec_list)
        vsec_array = (vsec_array - self.vmin) / (self.vmax - self.vmin)
        return vsec_array

    def _update_section_voltages(self, t_idx):
        if not self._vsec_recorded:
            raise RuntimeError(
                "Network must be simulated with"
                "`simulate_dipole(record_vsec='all')` before"
                "plotting voltages."
            )
        color_list = self.color_array[:, t_idx]
        for line, color in zip(self.ax.lines, color_list):
            line.set_color(color)

    def _init_network_plot(self):
        for cell_type in self.net.cell_types:
            gid_range = self.net.gid_ranges[cell_type]
            for gid_idx, gid in enumerate(gid_range):
                cell = self.net.cell_types[cell_type]

                pos = self.net.pos_dict[cell_type][gid_idx]
                pos = (float(pos[0]), float(pos[2]), float(pos[1]))

                cell.plot_morphology(
                    ax=self.ax,
                    show=False,
                    pos=pos,
                    xlim=self.xlim,
                    ylim=self.ylim,
                    zlim=self.zlim,
                )

    def _update_axes(self):
        self.ax.set_xlim(self._xlim)
        self.ax.set_ylim(self._ylim)
        self.ax.set_zlim(self._zlim)

        self.ax.view_init(self._elev, self._azim)

    def _update_colorbar(self):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mc

        fig = self.ax.get_figure()
        sm = plt.cm.ScalarMappable(
            cmap=self.voltage_colormap,
            norm=mc.Normalize(vmin=self.vmin, vmax=self.vmax),
        )
        self._cbar = fig.colorbar(sm, ax=self.ax)

    def export_movie(
        self,
        fname,
        fps=30,
        dpi=300,
        decim=10,
        interval=30,
        frame_start=0,
        frame_stop=None,
        writer="pillow",
    ):
        """Export movie of network activity

        Parameters
        ----------
        fname : str
            Filename of exported movie
        fps : int
            Frames per second, default: 30
        dpi : int
            Dots per inch, default: 300
        decim : int
            Decimation factor for frames, default: 10
        interval : int
            Delay between frames, default: 30
        frame_start : int
            Index of first frame, default: 0
        frame_stop : int | None
            Index of last frame, default: None
            If None, entire simulation is animated
        writer : str
            Movie writer, default: 'pillow'.
            Alternative movie writers can be found at
            https://matplotlib.org/stable/api/animation_api.html
        """
        import matplotlib.animation as animation

        if not self._vsec_recorded:
            raise RuntimeError(
                "Network must be simulated with"
                "`simulate_dipole(record_vsec='all')` before"
                "plotting voltages."
            )
        if frame_stop is None:
            frame_stop = len(self.times) - 1

        frames = np.arange(frame_start, frame_stop, decim)
        ani = animation.FuncAnimation(
            self.fig, self._set_time_idx, frames, interval=interval
        )

        writer = animation.writers[writer](fps=fps)
        ani.save(fname, writer=writer, dpi=dpi)
        return ani

    # Axis limits
    @property
    def xlim(self):
        return self._xlim

    @xlim.setter
    def xlim(self, xlim):
        _validate_type(xlim, tuple, "xlim")
        self._xlim = xlim
        self.ax.set_xlim(self._xlim)

    @property
    def ylim(self):
        return self._ylim

    @ylim.setter
    def ylim(self, ylim):
        _validate_type(ylim, tuple, "ylim")
        self._ylim = ylim
        self.ax.set_ylim(self._ylim)

    @property
    def zlim(self):
        return self._zlim

    @zlim.setter
    def zlim(self, zlim):
        _validate_type(zlim, tuple, "zlim")
        self._zlim = zlim
        self.ax.set_zlim(self._zlim)

    # Elevation and azimuth of 3D viewpoint
    @property
    def elev(self):
        return self._elev

    @elev.setter
    def elev(self, elev):
        _validate_type(elev, (int, float), "elev")
        self._elev = elev
        self.ax.view_init(self._elev, self._azim)

    @property
    def azim(self):
        return self._azim

    @azim.setter
    def azim(self, azim):
        _validate_type(azim, (int, float), "azim")
        self._azim = azim
        self.ax.view_init(self._elev, self._azim)

    # Minimum and maximum voltages
    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, vmin):
        _validate_type(vmin, (int, float), "vmin")
        self._vmin = vmin
        self.vsec_array = self._get_voltages()
        self.color_array = self._colormap(self.vsec_array)
        if self._colorbar:
            self._cbar.remove()
            self._update_colorbar()

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, vmax):
        _validate_type(vmax, (int, float), "vmax")
        self._vmax = vmax
        self.vsec_array = self._get_voltages()
        self.color_array = self._colormap(self.vsec_array)
        if self._colorbar:
            self._cbar.remove()
            self._update_colorbar()

    # Time and trial indices
    @property
    def trial_idx(self):
        return self._trial_idx

    @trial_idx.setter
    def trial_idx(self, trial_idx):
        _validate_type(trial_idx, int, "trial_idx")
        if not self._vsec_recorded:
            raise RuntimeError(
                "Network must be simulated with"
                "`simulate_dipole(record_vsec='all')` before"
                "setting `trial_idx`."
            )
        self._trial_idx = trial_idx
        self.vsec_array = self._get_voltages()
        self.color_array = self._colormap(self.vsec_array)
        self._update_section_voltages(self._time_idx)

    @property
    def time_idx(self):
        return self._time_idx

    @time_idx.setter
    def time_idx(self, time_idx):
        _validate_type(time_idx, (int, np.integer), "time_idx")
        if not self._vsec_recorded:
            raise RuntimeError(
                "Network must be simulated with"
                "`simulate_dipole(record_vsec='all')` before"
                "setting `time_idx`."
            )
        self._time_idx = time_idx
        self._update_section_voltages(self._time_idx)

    # Callable update function for making animations
    def _set_time_idx(self, time_idx):
        self.time_idx = time_idx

    # Background color and voltage colormaps
    @property
    def bg_color(self):
        return self._bg_color

    @bg_color.setter
    def bg_color(self, bg_color):
        self._bg_color = bg_color
        self.ax.set_facecolor(self._bg_color)

    @property
    def voltage_colormap(self):
        return self._voltage_colormap

    @voltage_colormap.setter
    def voltage_colormap(self, voltage_colormap):
        self._voltage_colormap = voltage_colormap
        self._colormap = self._colormaps[self._voltage_colormap]
        self.color_array = self._colormap(self.vsec_array)
        if self._colorbar:
            self._cbar.remove()
            self._update_colorbar()

    @property
    def colorbar(self):
        return self._colorbar

    @colorbar.setter
    def colorbar(self, colorbar):
        _validate_type(colorbar, bool, "colorbar")
        self._colorbar = colorbar
        if self._colorbar:
            # Remove old colorbar if already exists
            if self._cbar is not None:
                self._cbar.remove()
            self._update_colorbar()
        else:
            self._cbar.remove()
            self._cbar = None
