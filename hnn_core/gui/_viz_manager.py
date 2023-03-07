"""HNN GUI visualization management tool"""

# Authors: Huzi Cheng <hzcheng15@icloud.com>

import copy
import io
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (Box, Button, Dropdown, FloatText, HBox, Label, Layout,
                        Output, Tab, VBox, link)

from hnn_core.gui._logging import logger
from hnn_core.viz import plot_dipole

_fig_placeholder = 'Run simulation to add figures here.'

_plot_types = [
    'current dipole',
    'layer2 dipole',
    'layer5 dipole',
    'input histogram',
    'spikes',
    'PSD',
    'spectrogram',
    'network',
]

_no_overlay_plot_types = [
    'network',
    'spectrogram',
    'spikes',
    'input histogram',
]

_spectrogram_color_maps = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]

fig_templates = {
    "2row x 1col (1:3)": {
        "kwargs": "gridspec_kw={\"height_ratios\":[1,3]}",
        "mosaic": "00\n11",
    },
    "2row x 1col (1:1)": {
        "kwargs": "gridspec_kw={\"height_ratios\":[1,1]}",
        "mosaic": "00\n11",
    },
    "1row x 2col (1:1)": {
        "kwargs": "gridspec_kw={\"height_ratios\":[1,1]}",
        "mosaic": "01\n01",
    },
    "single figure": {
        "kwargs": "",
        "mosaic": "00\n00",
    },
    "2row x 2col (1:1)": {
        "kwargs": "gridspec_kw={\"height_ratios\":[1,1]}",
        "mosaic": "01\n23",
    },
}


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
    dpls_copied = copy.deepcopy(single_simulation['dpls'])
    net_copied = copy.deepcopy(single_simulation['net'])

    for dpl in dpls_copied:
        if plot_config['dipole_smooth'] > 0:
            dpl.smooth(plot_config['dipole_smooth']).scale(
                plot_config['dipole_scaling'])
        else:
            dpl.scale(plot_config['dipole_scaling'])

    if net_copied is None:
        print("No network data")
        return

    # Explicitly do this in case the
    # x and y axis are hidden after plotting some functions.
    ax.get_yaxis().set_visible(True)
    ax.get_xaxis().set_visible(True)
    if plot_type == 'spikes':
        if net_copied.cell_response:
            net_copied.cell_response.plot_spikes_raster(ax=ax, show=False)

    elif plot_type == 'input histogram':
        if net_copied.cell_response:
            net_copied.cell_response.plot_spikes_hist(ax=ax, show=False)

    elif plot_type == 'PSD':
        if len(dpls_copied) > 0:
            color = next(ax._get_lines.prop_cycler)['color']
            dpls_copied[0].plot_psd(fmin=0, fmax=50, ax=ax, color=color,
                                    label=sim_name, show=False)

    elif plot_type == 'spectrogram':
        if len(dpls_copied) > 0:
            min_f = 10.0
            max_f = plot_config['max_spectral_frequency']
            step_f = 1.0
            freqs = np.arange(min_f, max_f, step_f)
            n_cycles = freqs / 8.
            dpls_copied[0].plot_tfr_morlet(
                freqs,
                n_cycles=n_cycles,
                colormap=plot_config['spectrogram_cm'],
                ax=ax, colorbar_inside=True,
                show=False)

    elif 'dipole' in plot_type:
        if len(dpls_copied) > 0:
            color = next(ax._get_lines.prop_cycler)['color']
            if plot_type == 'current dipole':
                plot_dipole(dpls_copied,
                            ax=ax,
                            label=f"{sim_name}: average",
                            color=color,
                            average=True,
                            show=False)
            else:
                layer_namemap = {
                    "layer2": "L2",
                    "layer5": "L5",
                }
                plot_dipole(dpls_copied,
                            ax=ax,
                            label=f"{sim_name}: average",
                            color=color,
                            layer=layer_namemap[plot_type.split(" ")[0]],
                            average=True,
                            show=False)
        else:
            print("No dipole data")

    elif plot_type == 'network':
        if net_copied:
            with plt.ioff():
                _fig = plt.figure()
                _ax = _fig.add_subplot(111, projection='3d')
                net_copied.plot_cells(ax=_ax, show=False)
                io_buf = io.BytesIO()
                _fig.savefig(io_buf, format='raw')
                io_buf.seek(0)
                img_arr = np.reshape(np.frombuffer(io_buf.getvalue(),
                                     dtype=np.uint8),
                                     newshape=(int(_fig.bbox.bounds[3]),
                                     int(_fig.bbox.bounds[2]), -1))
                io_buf.close()
                _ = ax.imshow(img_arr)

    # set up alignment
    if plot_type not in ['network', 'PSD']:
        margin_x = 0
        max_x = max([dpl.times[-1] for dpl in dpls_copied])
        ax.set_xlim(left=-margin_x, right=max_x + margin_x)


def _static_rerender(widgets, fig, fig_idx):
    logger.debug('_static_re_render is called')
    figs_tabs = widgets['figs_tabs']
    titles = [
        figs_tabs.get_title(idx) for idx in range(len(figs_tabs.children))
    ]
    fig_tab_idx = titles.index(_idx2figname(fig_idx))
    fig_output = widgets['figs_tabs'].children[fig_tab_idx]
    fig_output.clear_output()
    with fig_output:
        fig.tight_layout()
        display(fig)


def _dynamic_rerender(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()


def _plot_on_axes(b, widgets_simulation, widgets_plot_type,
                  spectrogram_colormap_selection, dipole_smooth,
                  max_spectral_frequency, dipole_scaling, widgets, data,
                  fig_idx, fig, ax, existing_plots):
    sim_name = widgets_simulation.value
    plot_type = widgets_plot_type.value
    # disable add plots for types that do not support overlay
    if plot_type in _no_overlay_plot_types:
        b.disabled = True

    # freeze plot type
    widgets_plot_type.disabled = True

    single_simulation = data['simulations'][sim_name]

    plot_config = {
        "max_spectral_frequency": max_spectral_frequency.value,
        "dipole_scaling": dipole_scaling.value,
        "dipole_smooth": dipole_smooth.value,
        "spectrogram_cm": spectrogram_colormap_selection.value
    }

    _update_ax(fig, ax, single_simulation, sim_name, plot_type, plot_config)

    existing_plots.children = (*existing_plots.children,
                               Label(f"{sim_name}: {plot_type}"))
    if data['use_ipympl'] is False:
        _static_rerender(widgets, fig, fig_idx)
    else:
        _dynamic_rerender(fig)


def _clear_axis(b, widgets, data, fig_idx, fig, ax, widgets_plot_type,
                existing_plots, add_plot_button):
    ax.clear()

    # remove attached colorbar if exists
    if hasattr(fig, f'_cbar-ax-{id(ax)}'):
        getattr(fig, f'_cbar-ax-{id(ax)}').ax.remove()
        delattr(fig, f'_cbar-ax-{id(ax)}')

    ax.set_facecolor('w')
    ax.set_aspect('auto')
    widgets_plot_type.disabled = False
    add_plot_button.disabled = False
    existing_plots.children = ()
    if data['use_ipympl'] is False:
        _static_rerender(widgets, fig, fig_idx)
    else:
        _dynamic_rerender(fig)


def _get_ax_control(widgets, data, fig_idx, fig, ax):
    analysis_style = {'description_width': '200px'}
    layout = Layout(width="98%")
    simulation_names = tuple(data['simulations'].keys())
    if len(simulation_names) == 0:
        simulation_names = [
            "None",
        ]

    simulation_selection = Dropdown(
        options=simulation_names,
        value=simulation_names[-1],
        description='Simulation:',
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    plot_type_selection = Dropdown(
        options=_plot_types,
        value=_plot_types[0],
        description='Type:',
        disabled=False,
        layout=layout,
        style=analysis_style,
    )

    spectrogram_colormap_selection = Dropdown(
        description='Spectrogram Colormap:',
        options=[(cm, cm) for cm in _spectrogram_color_maps],
        value=_spectrogram_color_maps[0],
        layout=layout,
        style=analysis_style,
    )
    dipole_smooth = FloatText(value=30,
                              description='Dipole Smooth Window (ms):',
                              disabled=False,
                              layout=layout,
                              style=analysis_style)
    dipole_scaling = FloatText(value=3000,
                               description='Dipole Scaling:',
                               disabled=False,
                               layout=layout,
                               style=analysis_style)

    max_spectral_frequency = FloatText(
        value=100,
        description='Max Spectral Frequency (Hz):',
        disabled=False,
        layout=layout,
        style=analysis_style)

    existing_plots = VBox([])

    plot_button = Button(description='Add plot')
    clear_button = Button(description='Clear axis')

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
        ))

    plot_button.on_click(
        partial(
            _plot_on_axes,
            widgets_simulation=simulation_selection,
            widgets_plot_type=plot_type_selection,
            spectrogram_colormap_selection=spectrogram_colormap_selection,
            dipole_smooth=dipole_smooth,
            max_spectral_frequency=max_spectral_frequency,
            dipole_scaling=dipole_scaling,
            widgets=widgets,
            data=data,
            fig_idx=fig_idx,
            fig=fig,
            ax=ax,
            existing_plots=existing_plots,
        ))

    vbox = VBox([
        simulation_selection, plot_type_selection, dipole_smooth,
        dipole_scaling, max_spectral_frequency, spectrogram_colormap_selection,
        HBox(
            [plot_button, clear_button],
            layout=Layout(justify_content='space-between'),
        ), existing_plots], layout=Layout(width="98%"))

    return vbox


def _close_figure(b, widgets, data, fig_idx):
    fig_related_widgets = [widgets['figs_tabs'], widgets['axes_config_tabs']]
    for w_idx, tab in enumerate(fig_related_widgets):
        tab_children = list(tab.children)
        titles = [tab.get_title(idx) for idx in range(len(tab.children))]
        tab_idx = titles.index(_idx2figname(fig_idx))
        print(f"Del fig_idx={fig_idx}, fig_idx={fig_idx}")
        del tab_children[tab_idx], titles[tab_idx]

        tab.children = tuple(tab_children)
        [tab.set_title(idx, title) for idx, title in enumerate(titles)]

        if w_idx == 0:
            plt.close(data['figs'][fig_idx])
            del data['figs'][fig_idx]
            n_tabs = len(tab.children)
            for idx in range(n_tabs):
                _fig_idx = _figname2idx(tab.get_title(idx))
                assert _fig_idx in data['figs'].keys()

                tab.children[idx].clear_output()
                with tab.children[idx]:
                    display(data['figs'][_fig_idx].canvas)

        if n_tabs == 0:
            widgets['figs_output'].clear_output()
            with widgets['figs_output']:
                display(Label(_fig_placeholder))


def _add_axes_controls(widgets, data, fig, axd):
    fig_idx = data['fig_idx']['idx']

    controls = Tab()
    children = [
        _get_ax_control(widgets, data, fig_idx=fig_idx, fig=fig, ax=ax)
        for ax_key, ax in axd.items()
    ]
    controls.children = children
    for i in range(len(children)):
        controls.set_title(i, f'ax{i}')

    close_fig_button = Button(description=f'Close {_idx2figname(fig_idx)}',
                              button_style='danger',
                              icon='close',
                              layout=Layout(width="98%"))
    close_fig_button.on_click(
        partial(_close_figure, widgets=widgets, data=data, fig_idx=fig_idx))

    n_tabs = len(widgets['axes_config_tabs'].children)
    widgets['axes_config_tabs'].children = widgets[
        'axes_config_tabs'].children + (VBox([close_fig_button, controls]), )
    widgets['axes_config_tabs'].set_title(n_tabs, _idx2figname(fig_idx))


def _add_figure(b, widgets, data, scale=0.95, dpi=96):
    template_name = widgets['templates_dropdown'].value
    fig_idx = data['fig_idx']['idx']
    viz_output_layout = data['visualization_output']
    fig_outputs = Output()
    n_tabs = len(widgets['figs_tabs'].children)

    if n_tabs == 0:
        widgets['figs_output'].clear_output()
        with widgets['figs_output']:
            display(widgets['figs_tabs'])

    widgets['figs_tabs'].children = widgets['figs_tabs'].children + (
        fig_outputs, )
    widgets['figs_tabs'].set_title(n_tabs, _idx2figname(fig_idx))

    with fig_outputs:
        figsize = (scale * ((int(viz_output_layout.width[:-2]) - 10) / dpi),
                   scale * ((int(viz_output_layout.height[:-2]) - 10) / dpi))
        mosaic = fig_templates[template_name]['mosaic']
        kwargs = eval(f"dict({fig_templates[template_name]['kwargs']})")
        plt.ioff()
        fig, axd = plt.subplot_mosaic(mosaic,
                                      figsize=figsize,
                                      dpi=dpi,
                                      **kwargs)
        plt.ion()
        fig.tight_layout()
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False

        if data['use_ipympl'] is False:
            plt.show()
        else:
            display(fig.canvas)

    _add_axes_controls(widgets, data, fig=fig, axd=axd)

    data['figs'][fig_idx] = fig
    widgets['figs_tabs'].selected_index = n_tabs
    data['fig_idx']['idx'] += 1


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

    def __init__(self, gui_data, viz_layout):
        plt.close("all")
        self.viz_layout = viz_layout
        self.use_ipympl = 'ipympl' in matplotlib.get_backend()

        self.axes_config_output = Output()
        self.figs_output = Output()

        # widgets
        self.axes_config_tabs = Tab()
        self.figs_tabs = Tab()
        self.axes_config_tabs.selected_index = None
        self.figs_tabs.selected_index = None
        link(
            (self.axes_config_tabs, 'selected_index'),
            (self.figs_tabs, 'selected_index'),
        )

        template_names = list(fig_templates.keys())
        self.templates_dropdown = Dropdown(
            description='Layout template:',
            options=template_names,
            value=template_names[0],
            style={'description_width': 'initial'},
            layout=Layout(width="98%"))
        self.make_fig_button = Button(
            description='Make figure',
            button_style="primary",
            style={'button_color': self.viz_layout['theme_color']},
            layout=self.viz_layout['btn'])
        self.make_fig_button.on_click(self.add_figure)

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
            "templates_dropdown": self.templates_dropdown
        }

    @property
    def data(self):
        """Provides easy access to visualization-related data."""
        return {
            "use_ipympl": self.use_ipympl,
            "simulations": self.gui_data["simulation_data"],
            "fig_idx": self.fig_idx,
            "visualization_output": self.viz_layout['visualization_output'],
            "figs": self.figs
        }

    def reset_fig_config_tabs(self):
        """Reset the figure config tabs with most recent simulation data."""
        simulation_names = tuple(self.data['simulations'].keys())
        for tab in self.axes_config_tabs.children:
            controls = tab.children[1]
            for ax_control in controls.children:
                simulation_selection = ax_control.children[0]
                simulation_selection.options = simulation_names
        # recover the default layout
        self._simulate_switch_fig_template(list(fig_templates.keys())[0])

    def compose(self):
        """Compose widgets."""
        with self.axes_config_output:
            display(self.axes_config_tabs)
        with self.figs_output:
            display(Label(_fig_placeholder))

        fig_output_container = VBox(
            [self.figs_output], layout=self.viz_layout['visualization_window'])

        config_panel = VBox([
            Box(
                [
                    self.templates_dropdown,
                    self.make_fig_button,
                ],
                layout=Layout(
                    display='flex',
                    flex_flow='column',
                    align_items='stretch',
                ),
            ),
            Label("Figure config:"),
            self.axes_config_output,
        ])
        return config_panel, fig_output_container

    def add_figure(self, b=None):
        """Add a figure and corresponding config tabs to the dashboard.
        """
        _add_figure(None,
                    self.widgets,
                    self.data,
                    scale=0.97,
                    dpi=self.viz_layout['dpi'])

    def _simulate_add_fig(self):
        self.make_fig_button.click()

    def _simulate_switch_fig_template(self, template_name):
        assert template_name in fig_templates.keys(), "No such template"
        self.templates_dropdown.value = template_name

    def _simulate_delete_figure(self, fig_name):
        tab = self.axes_config_tabs
        titles = [tab.get_title(idx) for idx in range(len(tab.children))]
        assert fig_name in titles
        tab_idx = titles.index(fig_name)

        self.axes_config_tabs.selected_index = tab_idx
        close_button = self.axes_config_tabs.children[tab_idx].children[0]
        close_button.click()

    def _simulate_edit_figure(self, fig_name, ax_name, simulation_name,
                              plot_type, preprocessing_config, operation):
        """Manipulate a certain figure.

        Parameters
        ----------
            fig_name : str
                The figure name shown in the GUI, e.g., 'Figure 1'.
            ax_name : str
                Axis name shwon in the left side of GUI, like, 'ax0'.
            simulation_name : str
                The name of simulation you want to visualize
            plot_type : str
                Type of visualization.
            preprocessing_config : dict
                A dict of visualization preprocessing parameters. Allowed keys:
                `dipole_smooth`, `dipole_scaling`, `max_spectral_frequency`,
                `spectrogram_colormap_selection`. config could be empty: `{}`.
            operation : str
                `"plot"` if you want to plot and `"clear"` if you want to
                remove previously plotted visualizations.
        """
        assert simulation_name in self.data['simulations'].keys()
        assert plot_type in _plot_types
        assert operation in ("plot", "clear")

        tab = self.axes_config_tabs
        titles = [tab.get_title(idx) for idx in range(len(tab.children))]
        assert fig_name in titles, "No such figure"
        tab_idx = titles.index(fig_name)
        self.axes_config_tabs.selected_index = tab_idx

        ax_control_tabs = self.axes_config_tabs.children[tab_idx].children[1]
        ax_titles = [
            ax_control_tabs.get_title(idx)
            for idx in range(len(ax_control_tabs.children))
        ]
        assert ax_name in ax_titles, "No such axis"
        ax_idx = ax_titles.index(ax_name)
        ax_control_tabs.selected_index = ax_idx

        # ax config
        simulation_ctrl = ax_control_tabs.children[ax_idx].children[0]
        # return simulation_ctrl
        simulation_ctrl.value = simulation_name

        plot_type_ctrl = ax_control_tabs.children[ax_idx].children[1]
        plot_type_ctrl.value = plot_type

        config_name_idx = {
            "dipole_smooth": 2,
            "dipole_scaling": 3,
            "max_spectral_frequency": 4,
            "spectrogram_colormap_selection": 5,
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
