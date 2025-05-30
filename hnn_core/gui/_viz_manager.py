"""HNN GUI visualization management tool"""

# Authors: Huzi Cheng <hzcheng15@icloud.com>

import copy
import io
from functools import partial, wraps

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (
    Box,
    Button,
    Dropdown,
    BoundedFloatText,
    FloatText,
    HBox,
    Label,
    Layout,
    Output,
    Tab,
    VBox,
    link,
)

from hnn_core.dipole import average_dipoles, _rmse
from hnn_core.gui._logging import logger
from hnn_core.viz import plot_dipole

_fig_placeholder = "Run simulation to add figures here."

_plot_types = [
    "current dipole",
    "layer2/3 dipole",
    "layer5 dipole",
    "input histogram",
    "spikes",
    "spikes with dipoles",
    "PSD",
    "layer2/3 PSD",
    "layer5 PSD",
    "spectrogram",
    "network",
]

_no_overlay_plot_types = [
    "network",
    "spectrogram",
    "spikes",
    "spikes with dipoles",
    "input histogram",
]

_ext_data_disabled_plot_types = [
    "spikes",
    "spikes with dipoles",
    "input histogram",
    "network",
]

_spectrogram_color_maps = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]

fig_templates = {
    "[Blank] 2row x 1col (1:3)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 3]}},
        "mosaic": "00\n11",
    },
    "[Blank] 2row x 1col (1:1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 1]}},
        "mosaic": "00\n11",
    },
    "[Blank] 1row x 2col (1:1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 1]}},
        "mosaic": "01\n01",
    },
    "[Blank] single figure": {
        "kwargs": {"gridspec_kw": ""},
        "mosaic": "00\n00",
    },
    "[Blank] 2row x 2col (1:1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 1]}},
        "mosaic": "01\n23",
    },
}

data_templates = {
    "Drive-Dipole (2x1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 3]}},
        "mosaic": "00\n11",
        "ax_plots": [("ax0", "input histogram"), ("ax1", "current dipole")],
    },
    "Dipole Layers (3x1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 1, 1]}},
        "mosaic": "0\n1\n2",
        "ax_plots": [
            ("ax0", "layer2/3 dipole"),
            ("ax1", "layer5 dipole"),
            ("ax2", "current dipole"),
        ],
    },
    "Drive-Spikes (2x1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 3]}},
        "mosaic": "00\n11",
        "ax_plots": [("ax0", "input histogram"), ("ax1", "spikes")],
    },
    "Dipole-Spectrogram (2x1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 3]}},
        "mosaic": "00\n11",
        "ax_plots": [("ax0", "current dipole"), ("ax1", "spectrogram")],
    },
    "Dipole Layers-Spikes (1x1)": {
        "kwargs": {"gridspec_kw": ""},
        "mosaic": "00\n00",
        "ax_plots": [("ax0", "spikes with dipoles")],
    },
    "Drive-Dipole-Spectrogram (3x1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 1, 2]}},
        "mosaic": "0\n1\n2",
        "ax_plots": [
            ("ax0", "input histogram"),
            ("ax1", "current dipole"),
            ("ax2", "spectrogram"),
        ],
    },
    "PSD Layers (3x1)": {
        "kwargs": {"gridspec_kw": {"height_ratios": [1, 1, 1]}},
        "mosaic": "0\n1\n2",
        "ax_plots": [
            ("ax0", "layer2/3 PSD"),
            ("ax1", "layer5 PSD"),
            ("ax2", "PSD"),
        ],
    },
}


def check_sim_plot_types(new_sim_name, plot_type_selection, target_selection, data):
    if not _is_simulation(data["simulations"][new_sim_name]):
        plot_type_selection.options = [
            pt for pt in _plot_types if pt not in _ext_data_disabled_plot_types
        ]
    else:
        plot_type_selection.options = _plot_types
    # deal with target data
    all_possible_targets = list(data["simulations"].keys())
    all_possible_targets.remove(new_sim_name)
    target_selection.options = all_possible_targets + ["None"]
    target_selection.value = "None"


def _check_template_type_is_data_dependant(template_name):
    sim_data_options = list(data_templates.keys())
    return template_name in sim_data_options


def target_comparison_change(new_target_name, simulation_selection, data):
    """Triggered when the target data is turned on or changed."""
    pass


def plot_type_coupled_change(new_plot_type, target_data_selection):
    if new_plot_type != "current dipole":
        target_data_selection.disabled = True
    else:
        target_data_selection.disabled = False


def unlink_relink(attribute):
    """
    Decorator function to unlink widgets and re-link widgets.

    Unlinks linked widgets, runs the wrapped function, and relinks the widgets
    upon completion. To be used as a decorator on class methods. The class must
    have an attribute containing an ipywidgets/traitlets link object.

    Parameters
    ----------
    attribute: (str) The class attribute containing link object of ipywidgets
               widgets

    """

    def _unlink_relink(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            # Unlink the widgets using the provided link object
            link_attribute: link = getattr(self, attribute)
            link_attribute.unlink()

            # Call the original function
            result = f(self, *args, **kwargs)

            # Re-link the widgets
            link_attribute.link()

            return result

        return wrapper

    return _unlink_relink


def _idx2figname(idx):
    return f"Figure {idx}"


def _figname2idx(fname):
    return int(fname.split(" ")[-1])


def _update_ax(fig, ax, single_simulation, sim_name, plot_type, plot_config):
    """Refresh plots with simulation_data.

    Parameters
    ----------
    fig : Figure
        A matplotlib.figure.Figure object.
    ax : Axes
        matplotlib.axes.Axes
    single_simulation : dict
        A single simulation
    plot_type : str
        Type of subplots
    plot_config : dict
        A dict that specifies the preprocessing and style of plots.
    """
    # Make sure that visualization does not change the original data
    dpls_copied = copy.deepcopy(single_simulation["dpls"])
    net_copied = copy.deepcopy(single_simulation["net"])
    for dpl in dpls_copied:
        if plot_config["dipole_smooth"] > 0:
            dpl.smooth(plot_config["dipole_smooth"]).scale(
                plot_config["dipole_scaling"]
            )
        else:
            dpl.scale(plot_config["dipole_scaling"])

    if net_copied is None:
        assert plot_type not in _ext_data_disabled_plot_types

    # Explicitly do this in case the
    # x and y axis are hidden after plotting some functions.
    ax.get_yaxis().set_visible(True)
    ax.get_xaxis().set_visible(True)
    if plot_type == "spikes":
        if net_copied.cell_response:
            marker_size = plot_config["marker_size"]
            hide_spike_legend = plot_config["hide_spike_legend"]
            show_legend = True if hide_spike_legend == "False" else False
            net_copied.cell_response.plot_spikes_raster(
                ax=ax,
                show=False,
                show_legend=show_legend,
                marker_size=marker_size,
            )

    if plot_type == "spikes with dipoles":
        if net_copied.cell_response:
            marker_size = plot_config["marker_size"]
            hide_spike_legend = plot_config["hide_spike_legend"]
            show_legend = True if hide_spike_legend == "False" else False
            net_copied.cell_response.plot_spikes_raster(
                ax=ax,
                show=False,
                show_legend=show_legend,
                marker_size=marker_size,
                overlay_dipoles=True,
                dpl=dpls_copied,
            )

    elif plot_type == "input histogram":
        if net_copied.cell_response:
            net_copied.external_drives.keys()

            # initialize dictionary for drives, locations
            drive_locations = dict()
            drive_colors = dict()

            for name, drive in net_copied.external_drives.items():
                # remove all increments of default 'evdist' inputs
                if "evdist" in name:
                    if "evdist" not in drive_locations.keys():
                        drive_locations["evdist"] = drive["location"]
                        drive_colors["evdist"] = "g"
                # remove all increments of default 'evprox' inputs
                elif "evprox" in name:
                    if "evprox" not in drive_locations.keys():
                        drive_locations["evprox"] = drive["location"]
                        drive_colors["evprox"] = "r"
                else:
                    drive_locations[name] = drive["location"]
                    if drive["location"] == "proximal":
                        drive_colors[name] = "r"
                    elif drive["location"] == "distal":
                        drive_colors[name] = "g"

            # all drives to plot, excluding 'evdist' and 'evprox' increments
            all_drives = list(drive_locations.keys())

            # initialize list for distal drives
            distal_drives = list()

            for name, location in drive_locations.items():
                if location == "distal":
                    distal_drives.append(name)

            net_copied.cell_response.plot_spikes_hist(
                ax=ax,
                show=False,
                spike_types=all_drives,
                invert_spike_types=distal_drives,
                color=drive_colors,
            )

    elif plot_type == "PSD":
        if len(dpls_copied) > 0:
            min_f = plot_config["min_spectral_frequency"]
            max_f = plot_config["max_spectral_frequency"]
            color = ax._get_lines.get_next_color()
            label = sim_name + " (Aggregate)"
            dpls_copied[0].plot_psd(
                fmin=min_f, fmax=max_f, color=color, label=label, ax=ax, show=False
            )

    elif plot_type == "layer2/3 PSD":
        if len(dpls_copied) > 0:
            min_f = plot_config["min_spectral_frequency"]
            max_f = plot_config["max_spectral_frequency"]
            color = ax._get_lines.get_next_color()
            label = sim_name + " (Layer 2/3)"
            dpls_copied[0].plot_psd(
                fmin=min_f,
                fmax=max_f,
                layer="L2",
                color=color,
                label=label,
                ax=ax,
                show=False,
            )

    elif plot_type == "layer5 PSD":
        if len(dpls_copied) > 0:
            min_f = plot_config["min_spectral_frequency"]
            max_f = plot_config["max_spectral_frequency"]
            color = ax._get_lines.get_next_color()
            label = sim_name + " (Layer 5)"
            dpls_copied[0].plot_psd(
                fmin=min_f,
                fmax=max_f,
                layer="L5",
                color=color,
                label=label,
                ax=ax,
                show=False,
            )

    elif plot_type == "spectrogram":
        if len(dpls_copied) > 0:
            min_f = plot_config["min_spectral_frequency"]
            max_f = plot_config["max_spectral_frequency"]
            step_f = 1.0
            if min_f > max_f:
                step_f = -1
            freqs = np.arange(min_f, max_f, step_f)
            n_cycles = freqs / 2.0

            try:
                dpls_copied[0].plot_tfr_morlet(
                    freqs,
                    n_cycles=n_cycles,
                    colormap=plot_config["spectrogram_cm"],
                    ax=ax,
                    colorbar_inside=True,
                    show=False,
                )

            except ValueError as ex:
                if str(ex) == (
                    "At least one of the wavelets is longer than "
                    "the signal. Use a longer signal or shorter "
                    "wavelets."
                ):
                    logger.error(
                        "At least one of the wavelets is "
                        "longer than the signal. Use a longer signal "
                        "or shorter wavelets. No spectrogram will be "
                        "plotted."
                    )

    elif "dipole" in plot_type:
        if len(dpls_copied) > 0:
            if len(dpls_copied) > 1:
                label = f"{sim_name}: average"
            else:
                label = sim_name

            color = ax._get_lines.get_next_color()
            if plot_type == "current dipole":
                plot_dipole(
                    dpls_copied,
                    ax=ax,
                    label=label,
                    color=color,
                    average=True,
                    show=False,
                )
            else:
                layer_namemap = {
                    "layer2/3": "L2",
                    "layer5": "L5",
                }
                plot_dipole(
                    dpls_copied,
                    ax=ax,
                    label=label,
                    color=color,
                    layer=layer_namemap[plot_type.split(" ")[0]],
                    average=True,
                    show=False,
                )
        else:
            print("No dipole data")

    elif plot_type == "network":
        if net_copied:
            with plt.ioff():
                _fig = plt.figure()
                _ax = _fig.add_subplot(111, projection="3d")
                net_copied.plot_cells(ax=_ax, show=False)
                io_buf = io.BytesIO()
                _fig.savefig(io_buf, format="raw")
                io_buf.seek(0)
                img_arr = np.reshape(
                    np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                    newshape=(int(_fig.bbox.bounds[3]), int(_fig.bbox.bounds[2]), -1),
                )
                io_buf.close()
                _ = ax.imshow(img_arr)

    # set up alignment of plots which do NOT use dipole's time as their x-axis
    custom_x_axes = [
        "network",
        "PSD",
        "layer2/3 PSD",
        "layer5 PSD",
    ]
    if plot_type not in custom_x_axes:
        margin_x = 0
        max_x = max([dpl.times[-1] for dpl in dpls_copied])
        ax.set_xlim(left=-margin_x, right=max_x + margin_x)

    return dpls_copied


def _static_rerender(widgets, fig, fig_idx):
    logger.debug("_static_re_render is called")
    figs_tabs = widgets["figs_tabs"]
    titles = figs_tabs.titles
    fig_tab_idx = titles.index(_idx2figname(fig_idx))
    fig_output = widgets["figs_tabs"].children[fig_tab_idx]
    fig_output.clear_output()
    with fig_output:
        display(fig)


def _dynamic_rerender(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()


def _avg_dipole_check(dpls):
    """Check for averaged dipole, else average the trials"""
    # Check if there is an averaged dipole already
    if not dpls:
        return None

    avg_dpls = [d for d in dpls if d.nave > 1]
    if avg_dpls:
        dpl = avg_dpls[0]
    else:
        dpl = average_dipoles(dpls)
    return dpl


def _plot_on_axes(
    b,
    simulations_widget,
    widgets_plot_type,
    data_widget,
    spectrogram_colormap_selection,
    hide_spike_legend,
    marker_size,
    min_spectral_frequency,
    max_spectral_frequency,
    dipole_smooth,
    dipole_scaling,
    data_smooth,
    data_scaling,
    widgets,
    data,
    fig_idx,
    fig,
    ax,
    existing_plots,
):
    """Plotting different types of data on the given axes.

    Now this function is also responsible for comparing multiple simulations,
    or simulations vs. experimental data.

    Parameters
    ----------
    b : ipywidgets.Button
    widgets_simulation : ipywidgets.Dropdown
        A dropdown widget that contains all the simulation names.
    widgets_plot_type : ipywidgets.Dropdown
        A dropdown widget that contains all the plot types.
    target_simulations : ipywidgets.Dropdown
        The target data we want to compare with. Note that this could be 'None'
    spectrogram_colormap_selection : ipywidgets.Dropdown
        A dropdown widget that contains all the colormaps for spectrogram.
    hide_spike_legend : ipywidgets.Dropdown
        A dropdown widget that specifies whether to hide the legend for
        the spikes raster plot.
    marker_size : ipywidgets.BoundedFloatText
        A widget that specifies the marker size for the raster plot.
    dipole_smooth : ipywidgets.FloatText
        A textfield widget that specifies the smoothing window size.
    min_spectral_frequency : ipywidgets.FloatText
        A textfield that specifies the minimum frequency for spectrogram plot.
    max_spectral_frequency : ipywidgets.FloatText
        A textfield that specifies the max frequency for spectrogram plot.
    dipole_scaling : ipywidgets.FloatText
        A textfield that specifies the scaling factor for dipole object.
    widgets : dict
        A dict that contains all the widgets.
    data : dict
        A dict that contains all the simulation data. Can be accessed by names
        specified in widgets_simulation and target_simulations weidgets.
    fig_idx : int
        The index of the figure we want to plot on.
    fig : matplotlib.figure.Figure
        The figure we want to plot on.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes we want to operate.
    existing_plots : ipywidgets.VBox
        A VBox widget that contains all the existing plots.
    """
    sim_name = simulations_widget.value
    plot_type = widgets_plot_type.value
    # disable add plots for types that do not support overlay
    if plot_type in _no_overlay_plot_types:
        b.disabled = True

    # freeze plot type
    widgets_plot_type.disabled = True

    single_simulation = data["simulations"][sim_name]
    simulation_plot_config = {
        "dipole_scaling": dipole_scaling.value,
        "dipole_smooth": dipole_smooth.value,
        "min_spectral_frequency": min_spectral_frequency.value,
        "max_spectral_frequency": max_spectral_frequency.value,
        "spectrogram_cm": spectrogram_colormap_selection.value,
        "hide_spike_legend": hide_spike_legend.value,
        "marker_size": marker_size.value,
    }

    dpls_processed = _update_ax(
        fig, ax, single_simulation, sim_name, plot_type, simulation_plot_config
    )

    # If target_simulations is not None and we are plotting a dipole,
    # we need to plot the target dipole as well.
    if (
        data_widget.value in data["simulations"].keys()
        and plot_type == "current dipole"
    ):
        target_sim_name = data_widget.value
        target_sim = data["simulations"][target_sim_name]
        data_plot_config = {
            "dipole_scaling": data_scaling.value,
            "dipole_smooth": data_smooth.value,
            "min_spectral_frequency": min_spectral_frequency.value,
            "max_spectral_frequency": max_spectral_frequency.value,
            "spectrogram_cm": spectrogram_colormap_selection.value,
            "hide_spike_legend": hide_spike_legend.value,
            "marker_size": marker_size.value,
        }

        # plot the target dipole.
        target_dpl_processed = _update_ax(
            fig, ax, target_sim, target_sim_name, plot_type, data_plot_config
        )[0]  # we assume there is only one dipole.

        # calculate the RMSE between the two dipoles.
        t0 = 0.0
        tstop = dpls_processed[-1].times[-1]
        if len(dpls_processed) > 1:
            dpl = _avg_dipole_check(dpls_processed)
        else:
            dpl = dpls_processed
        rmse = _rmse(dpl, target_dpl_processed, t0, tstop)
        annotation_text = f"RMSE({sim_name}, {target_sim_name}): {rmse:.4f}"

        # find subplot's annotation
        annotation = next(
            (child for child in ax.get_children() if isinstance(child, plt.Annotation)),
            None,
        )

        # if the subplot already has an annotation, update its text.
        # Otherwise, create a new one.
        if annotation is not None:
            annotation.set_text(annotation_text)
        else:
            ax.annotate(
                annotation_text,
                xy=(0.95, 0.05),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=12,
            )

        rmse_logger_text = (
            f"RMSE {rmse:.4f} ("
            f"{sim_name} smooth:{dipole_smooth.value} "
            f"scale:{dipole_scaling.value} \n"
            f"{target_sim_name} smooth:{data_smooth.value} "
            f"scale:{data_scaling.value})"
        )

        logger.info(rmse_logger_text)

    existing_plots.children = (
        *existing_plots.children,
        Label(f"{sim_name}: {plot_type}"),
    )
    if data["use_ipympl"] is False:
        _static_rerender(widgets, fig, fig_idx)
    else:
        _dynamic_rerender(fig)


def _clear_axis(
    b,
    widgets,
    data,
    fig_idx,
    fig,
    ax,
    widgets_plot_type,
    existing_plots,
    add_plot_button,
):
    ax.clear()

    # Remove "plot_spikes_hist"'s inverted second axes object, if exists, and
    # if the axis you are clearing is the spike histogram
    if ax._label == "Spike histogram":
        for axis in fig.axes:
            if axis._label == "Inverted spike histogram":
                axis.remove()

    # remove attached colorbar if exists
    if hasattr(fig, f"_cbar-ax-{id(ax)}"):
        getattr(fig, f"_cbar-ax-{id(ax)}").ax.remove()
        delattr(fig, f"_cbar-ax-{id(ax)}")

    ax.set_facecolor("w")
    ax.set_aspect("auto")
    widgets_plot_type.disabled = False
    add_plot_button.disabled = False
    existing_plots.children = ()
    if data["use_ipympl"] is False:
        _static_rerender(widgets, fig, fig_idx)
    else:
        _dynamic_rerender(fig)


def _get_ax_control(widgets, data, fig_default_params, fig_idx, fig, ax):
    analysis_style = {"description_width": "200px"}
    layout = Layout(width="98%")
    simulation_names = tuple(data["simulations"].keys())
    sim_index = 0
    default_smoothing = fig_default_params["default_smoothing"]
    default_scaling = fig_default_params["default_scaling"]
    default_min_frequency = fig_default_params["default_min_frequency"]
    default_max_frequency = fig_default_params["default_max_frequency"]
    if not simulation_names:
        simulation_names = ("None",)
    else:
        # Find the last simulation with a non-None 'net'
        sim_index = next(
            (
                idx
                for idx, sim_name in reversed(list(enumerate(simulation_names)))
                if _is_simulation(data["simulations"][sim_name])
            ),
            0,  # Default value if no such simulation is found
        )

    simulation_selection = Dropdown(
        options=simulation_names,
        value=simulation_names[sim_index],
        description="Simulation Data:",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    plot_type_selection = Dropdown(
        options=_plot_types,
        value=_plot_types[0],
        description="Type:",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    tagert_names = simulation_names[:-1]
    if len(simulation_names) > 1:
        tagert_names = simulation_names[1:]

    target_data_selection = Dropdown(
        options=tagert_names + ("None",),
        value="None",
        description="Data to Compare:",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    # This will check the sim plot types dropdown available options
    # for the specific sim name in the simulation_selection dropdown options
    check_sim_plot_types(
        simulation_names[sim_index], plot_type_selection, target_data_selection, data
    )

    spectrogram_colormap_selection = Dropdown(
        description="Spectrogram Colormap:",
        options=[(cm, cm) for cm in _spectrogram_color_maps],
        value=_spectrogram_color_maps[0],
        layout=layout,
        style=analysis_style,
    )
    simulation_dipole_smooth = FloatText(
        value=default_smoothing,
        description="Dipole Smooth Window (ms):",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    simulation_dipole_scaling = FloatText(
        value=default_scaling,
        description="Simulation Dipole Scaling:",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    data_dipole_smooth = FloatText(
        value=0,
        description="Data Smooth Window (ms):",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    data_dipole_scaling = FloatText(
        value=1,
        description="Data Dipole Scaling:",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    min_spectral_frequency = BoundedFloatText(
        value=default_min_frequency,
        min=0.1,
        max=1000,
        description="Min Spectral Frequency (Hz):",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    max_spectral_frequency = BoundedFloatText(
        value=default_max_frequency,
        min=0.1,
        max=1000,
        description="Max Spectral Frequency (Hz):",
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    hide_spike_legend = Dropdown(
        description="Hide Raster Plot Legend:",
        options=["True", "False"],
        value="False",
        layout=layout,
        style=analysis_style,
    )

    marker_size = BoundedFloatText(
        value=5,
        min=0.01,
        max=15,
        description="Raster Plot Marker Size:",
        layout=layout,
        style=analysis_style,
    )

    existing_plots = VBox([])

    plot_button = Button(description="Add plot")
    clear_button = Button(description="Clear axis")

    def _on_sim_data_change(new_sim_name):
        return check_sim_plot_types(
            new_sim_name.new, plot_type_selection, target_data_selection, data
        )

    def _on_target_comparison_change(new_target_name):
        return target_comparison_change(new_target_name, simulation_selection, data)

    def _on_plot_type_change(new_plot_type):
        return plot_type_coupled_change(new_plot_type.new, target_data_selection)

    simulation_selection.observe(_on_sim_data_change, "value")
    target_data_selection.observe(_on_target_comparison_change, "value")
    plot_type_selection.observe(_on_plot_type_change, "value")

    clear_button.on_click(
        partial(
            _clear_axis,
            widgets=widgets,
            data=data,
            fig_idx=fig_idx,
            fig=fig,
            ax=ax,
            widgets_plot_type=plot_type_selection,
            existing_plots=existing_plots,
            add_plot_button=plot_button,
        )
    )

    plot_button.on_click(
        partial(
            _plot_on_axes,
            simulations_widget=simulation_selection,
            widgets_plot_type=plot_type_selection,
            data_widget=target_data_selection,
            spectrogram_colormap_selection=spectrogram_colormap_selection,
            hide_spike_legend=hide_spike_legend,
            marker_size=marker_size,
            min_spectral_frequency=min_spectral_frequency,
            max_spectral_frequency=max_spectral_frequency,
            dipole_smooth=simulation_dipole_smooth,
            dipole_scaling=simulation_dipole_scaling,
            data_smooth=data_dipole_smooth,
            data_scaling=data_dipole_scaling,
            widgets=widgets,
            data=data,
            fig_idx=fig_idx,
            fig=fig,
            ax=ax,
            existing_plots=existing_plots,
        )
    )

    vbox = VBox(
        [
            plot_type_selection,
            simulation_selection,
            simulation_dipole_smooth,
            simulation_dipole_scaling,
            target_data_selection,
            data_dipole_smooth,
            data_dipole_scaling,
            min_spectral_frequency,
            max_spectral_frequency,
            spectrogram_colormap_selection,
            hide_spike_legend,
            marker_size,
            HBox(
                [plot_button, clear_button],
                layout=Layout(justify_content="space-between"),
            ),
            existing_plots,
        ],
        layout=Layout(width="98%"),
    )

    return vbox


def _close_figure(b, widgets, data, fig_idx):
    fig_related_widgets = [widgets["figs_tabs"], widgets["axes_config_tabs"]]
    for w_idx, tab in enumerate(fig_related_widgets):
        # Get tab object's list of children and their titles
        tab_children = list(tab.children)
        titles = list(tab.titles)
        # Get the index based on the title
        tab_idx = titles.index(_idx2figname(fig_idx))
        # Remove the child and title specified
        print(f"Del fig_idx={fig_idx}, fig_idx={fig_idx}")
        tab_children.pop(tab_idx)
        titles.pop(tab_idx)
        # Reset children and titles of the tab object
        tab.children = tab_children
        tab.titles = titles

        # If the figure tab group...
        if w_idx == 0:
            # Close figure and delete the data
            plt.close(data["figs"][fig_idx])
            data["figs"].pop(fig_idx)
            # Redisplay the remaining children
            n_tabs = len(tab.children)
            for idx in range(n_tabs):
                _fig_idx = _figname2idx(tab.get_title(idx))
                assert _fig_idx in data["figs"].keys()

                tab.children[idx].clear_output()
                with tab.children[idx]:
                    display(data["figs"][_fig_idx].canvas)

            # If all children have been deleted display the placeholder
            if n_tabs == 0:
                widgets["figs_output"].clear_output()
                with widgets["figs_output"]:
                    display(Label(_fig_placeholder))


def _add_axes_controls(widgets, data, fig_default_params, fig, axd):
    fig_idx = data["fig_idx"]["idx"]

    controls = Tab()
    children = [
        _get_ax_control(
            widgets, data, fig_default_params, fig_idx=fig_idx, fig=fig, ax=ax
        )
        for ax_key, ax in axd.items()
    ]
    controls.children = children
    for i in range(len(children)):
        controls.set_title(i, f"ax{i}")

    close_fig_button = Button(
        description=f"Close {_idx2figname(fig_idx)}",
        button_style="danger",
        icon="close",
        layout=Layout(width="98%"),
    )
    close_fig_button.on_click(
        partial(_close_figure, widgets=widgets, data=data, fig_idx=fig_idx)
    )

    n_tabs = len(widgets["axes_config_tabs"].children)
    widgets["axes_config_tabs"].children = widgets["axes_config_tabs"].children + (
        VBox([close_fig_button, controls]),
    )
    widgets["axes_config_tabs"].set_title(n_tabs, _idx2figname(fig_idx))


def _add_figure(
    b, widgets, data, fig_default_params, template_type, scale=0.95, dpi=96
):
    fig_idx = data["fig_idx"]["idx"]
    viz_output_layout = data["visualization_output"]
    fig_outputs = Output()
    n_tabs = len(widgets["figs_tabs"].children)

    if n_tabs == 0:
        widgets["figs_output"].clear_output()
        with widgets["figs_output"]:
            display(widgets["figs_tabs"])

    widgets["figs_tabs"].children = [s for s in widgets["figs_tabs"].children] + [
        fig_outputs
    ]
    widgets["figs_tabs"].set_title(n_tabs, _idx2figname(fig_idx))

    with fig_outputs:
        figsize = (
            scale * ((int(viz_output_layout.width[:-2]) - 10) / dpi),
            scale * ((int(viz_output_layout.height[:-2]) - 10) / dpi),
        )
        mosaic = template_type["mosaic"]
        kwargs = template_type["kwargs"]
        with plt.ioff():
            fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
            axd = fig.subplot_mosaic(mosaic, **kwargs)
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False

        if data["use_ipympl"] is False:
            plt.show()
        else:
            display(fig.canvas)

    _add_axes_controls(widgets, data, fig_default_params, fig=fig, axd=axd)

    data["figs"][fig_idx] = fig
    widgets["figs_tabs"].selected_index = n_tabs
    widgets["axes_config_tabs"].selected_index = n_tabs
    data["fig_idx"]["idx"] += 1


def _postprocess_template(template_name, fig, idx, use_ipympl=True, widgets=None):
    """Post-processes and re-renders plot templates with determined styles

    Templates are constructed on panel-by-panel basis. If adjustments need to
    be made based on information other plots in the figure, it is adjusted with
    this function. For example, L2 and L5 dipole plots should have the same
    y-axis range.
    """
    if template_name not in ["Dipole Layers (3x1)"]:
        return

    if template_name == "Dipole Layers (3x1)":
        # Make the L2, L5, and aggregate plots use the same y-range
        y_lims_list = [ax.get_ylim() for ax in fig.axes]
        y_lims = (np.min(y_lims_list), np.max(y_lims_list))
        [ax.set_ylim(y_lims) for ax in fig.axes]

    # Re-render
    if not use_ipympl:
        _static_rerender(widgets, fig, idx)
    else:
        _dynamic_rerender(fig)


class _VizManager:
    """GUI visualization panel manager class.

    Parameters
    ----------
    gui_data : dict
        A dict containing all simulation data
    viz_layout : dict
        A dict about visualization layout specs

    Attributes
    ----------
    widgets : dict
        A dict of widget references
    data : dict
        A dict of external simulation data object
    """

    def __init__(self, gui_data, viz_layout, fig_default_params):
        plt.close("all")
        self.viz_layout = viz_layout
        self.fig_default_params = fig_default_params
        self.use_ipympl = "ipympl" in matplotlib.get_backend()

        self.axes_config_output = Output()
        self.figs_output = Output()

        # widgets
        self.axes_config_tabs = Tab()
        self.figs_tabs = Tab()
        self.axes_config_tabs.selected_index = None
        self.figs_tabs.selected_index = None
        self.figs_config_tab_link = link(
            (self.axes_config_tabs, "selected_index"),
            (self.figs_tabs, "selected_index"),
        )

        template_names = list(data_templates.keys())
        template_names.extend(list(fig_templates.keys()))
        self.templates_dropdown = Dropdown(
            description="Layout template:",
            options=template_names,
            value=template_names[0],
            style={"description_width": "initial"},
            layout=Layout(width="98%"),
        )
        self.templates_dropdown.observe(self._layout_template_change, "value")

        self.make_fig_button = Button(
            description="Make figure",
            button_style="primary",
            style={"button_color": self.viz_layout["theme_color"]},
            layout=self.viz_layout["btn"],
        )
        self.make_fig_button.on_click(self.add_figure)

        self.datasets_dropdown = Dropdown(
            description="Dataset:",
            options=[],
            value=None,
            style={"description_width": "initial"},
            layout=Layout(width="98%"),
        )

        # data
        self.fig_idx = {"idx": 1}
        self.figs = {}
        self.gui_data = gui_data

    @property
    def widgets(self):
        return {
            "figs_output": self.figs_output,
            "axes_config_tabs": self.axes_config_tabs,
            "figs_tabs": self.figs_tabs,
            "templates_dropdown": self.templates_dropdown,
            "dataset_dropdown": self.datasets_dropdown,
        }

    @property
    def data(self):
        """Provides easy access to visualization-related data."""
        return {
            "use_ipympl": self.use_ipympl,
            "simulations": self.gui_data["simulation_data"],
            "fig_idx": self.fig_idx,
            "visualization_output": self.viz_layout["visualization_output"],
            "figs": self.figs,
        }

    def reset_fig_config_tabs(self, template_name=None):
        """Reset the figure config tabs with most recent simulation data."""
        simulation_names = tuple(self.data["simulations"].keys())
        for tab in self.axes_config_tabs.children:
            controls = tab.children[1]
            for ax_control in controls.children:
                # Update the options for the simulation data selection dropdown
                simulation_data_selection = ax_control.children[1]
                simulation_data_selection.options = simulation_names

                # Update the options for the data to compare dropdown
                simulation_to_compare = ax_control.children[4]
                simulation_to_compare.options = simulation_names

        # recover the default layout
        if template_name is None:
            template_name = list(fig_templates.keys())[0]
        self._simulate_switch_fig_template(template_name)

    def compose(self):
        """Compose widgets."""
        with self.axes_config_output:
            display(self.axes_config_tabs)
        with self.figs_output:
            display(Label(_fig_placeholder))

        fig_output_container = VBox(
            [self.figs_output], layout=self.viz_layout["visualization_window"]
        )

        config_panel = VBox(
            [
                Box(
                    [
                        self.templates_dropdown,
                        self.datasets_dropdown,
                        self.make_fig_button,
                    ],
                    layout=Layout(
                        display="flex",
                        flex_flow="column",
                        align_items="stretch",
                    ),
                ),
                Label("Figure config:"),
                self.axes_config_output,
            ]
        )
        return config_panel, fig_output_container

    def _layout_template_change(self, template_type):
        # check if plot set type requires loaded sim-data
        if _check_template_type_is_data_dependant(template_type.new):
            # Add only simualated data
            sim_names = [
                simulations
                for simulations, sim_name in self.data["simulations"].items()
                if sim_name["net"] is not None
            ]

            if len(sim_names) == 0:
                sim_names = [" "]

            self.datasets_dropdown.options = sim_names
            self.datasets_dropdown.value = sim_names[0]
            # show list of simulated to gui dropdown
            self.datasets_dropdown.layout.visibility = "visible"
        else:
            # hide sim-data dropdown
            self.datasets_dropdown.layout.visibility = "hidden"

    @unlink_relink(attribute="figs_config_tab_link")
    def add_figure(self, b=None):
        """Add a figure and corresponding config tabs to the dashboard."""
        if len(self.data["simulations"]) == 0:
            logger.error("No data has been loaded")
            return

        template_name = self.widgets["templates_dropdown"].value
        is_data_template = _check_template_type_is_data_dependant(template_name)
        if is_data_template:
            sim_name = self.widgets["dataset_dropdown"].value
            if sim_name not in self.data["simulations"]:
                logger.error("No simulation data has been loaded")
                return

        # Use data_templates dictionary if it's a data dependent layout
        template_type = (
            data_templates[template_name]
            if is_data_template
            else fig_templates[template_name]
        )

        # Add empty figure according to template arguments
        _add_figure(
            None,
            self.widgets,
            self.data,
            self.fig_default_params,
            template_type,
            scale=0.97,
            dpi=self.viz_layout["dpi"],
        )

        # Plot data if it is a data-dependent template
        if is_data_template:
            fig_name = _idx2figname(self.data["fig_idx"]["idx"] - 1)
            # get figs per axis
            ax_plots = data_templates[template_name]["ax_plots"]
            for ax_name, plot_type in ax_plots:
                # paint fig in axis
                self._simulate_edit_figure(
                    fig_name, ax_name, sim_name, plot_type, {}, "plot"
                )
            # template post-processing
            fig_key = self.data["fig_idx"]["idx"] - 1
            _postprocess_template(
                template_name,
                fig=self.figs[fig_key],
                idx=fig_key,
                use_ipympl=self.use_ipympl,
                widgets=self.widgets,
            )

            logger.info(
                f"Figure {template_name} for simulation {sim_name} has been created"
            )

    def _simulate_add_fig(self):
        self.make_fig_button.click()

    def _simulate_switch_fig_template(self, template_name):
        assert template_name in fig_templates.keys() or data_templates.keys(), (
            "No such template"
        )
        self.templates_dropdown.value = template_name

    def _simulate_delete_figure(self, fig_name):
        tab = self.axes_config_tabs
        titles = tab.titles
        assert fig_name in titles
        tab_idx = titles.index(fig_name)

        self.axes_config_tabs.selected_index = tab_idx
        close_button = self.axes_config_tabs.children[tab_idx].children[0]
        close_button.click()

    def _simulate_edit_figure(
        self,
        fig_name,
        ax_name,
        simulation_name,
        plot_type,
        preprocessing_config,
        operation,
    ):
        """Manipulate a certain figure.

        Parameters
        ----------
            fig_name : str
                The figure name shown in the GUI, e.g., 'Figure 1'.
            ax_name : str
                Axis name shown in the left side of GUI, like, 'ax0'.
            simulation_name : str
                The name of simulation you want to visualize
            plot_type : str
                Type of visualization.
            preprocessing_config : dict
                A dict of visualization preprocessing parameters. Allowed keys:
                `dipole_smooth`, `dipole_scaling`,
                `data_to_compare`, `data_smooth`, `data_scaling`
                `min_spectral_frequency`, `max_spectral_frequency`,
                `spectrogram_colormap_selection`, `hide_spike_legend,
                `marker_size`.
                config could be empty: `{}`.
            operation : str
                `"plot"` if you want to plot and `"clear"` if you want to
                remove previously plotted visualizations.
        """
        assert simulation_name in self.data["simulations"].keys()
        assert plot_type in _plot_types
        assert operation in ("plot", "clear")

        # Select the figure tab
        tab = self.axes_config_tabs
        titles = tab.titles
        assert fig_name in titles, "No such figure"
        tab_idx = titles.index(fig_name)
        self.axes_config_tabs.selected_index = tab_idx

        # Select the figure panel/ax tab
        ax_control_tabs = self.axes_config_tabs.children[tab_idx].children[1]
        ax_titles = ax_control_tabs.titles
        assert ax_name in ax_titles, "No such axis"
        ax_idx = ax_titles.index(ax_name)
        ax_control_tabs.selected_index = ax_idx

        # Select the simulation
        simulation_selector = ax_control_tabs.children[ax_idx].children[1]
        simulation_selector.value = simulation_name

        # Select the plot type
        plot_type_selector = ax_control_tabs.children[ax_idx].children[0]
        plot_type_selector.value = plot_type

        # Set the plot configurations
        config_name_idx = {
            "dipole_smooth": 2,
            "dipole_scaling": 3,
            "data_to_compare": 4,
            "data_smooth": 5,
            "data_scaling": 6,
            "min_spectral_frequency": 7,
            "max_spectral_frequency": 8,
            "spectrogram_colormap_selection": 9,
            "hide_spike_legend": 10,
            "marker_size": 11,
        }
        for conf_key, conf_val in preprocessing_config.items():
            assert conf_key in config_name_idx.keys()
            idx = config_name_idx[conf_key]
            conf_widget = ax_control_tabs.children[ax_idx].children[idx]
            conf_widget.value = conf_val

        buttons = ax_control_tabs.children[ax_idx].children[-2]
        if operation == "plot":
            buttons.children[0].click()
        elif operation == "clear":
            buttons.children[1].click()


def _is_simulation(data):
    """Determines if saved data is a simulation."""
    return data["net"] is not None
