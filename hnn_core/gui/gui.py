"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>

import codecs
import logging
import multiprocessing
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display
from ipywidgets import (HTML, Accordion, AppLayout, BoundedFloatText,
                        BoundedIntText, Button, Dropdown, FileUpload,
                        FloatLogSlider, FloatText, GridspecLayout, HBox,
                        IntText, Layout, Output, RadioButtons, Tab, Text, VBox,
                        interactive_output, link)

import hnn_core
from hnn_core import (JoblibBackend, MPIBackend, jones_2009_model, read_params,
                      simulate_dipole)
from hnn_core.params import (_extract_drive_specs_from_hnn_params, _read_json,
                             _read_legacy_params)
from hnn_core.viz import plot_dipole


class HNNGUI:
    """HNN GUI class

    Parameters
    ----------
    theme_color : str
        The theme color of the whole dashboard.
    log_window_height: str
        The height of the log output window (in pixel).
    visualization_window_width: str
        The width of the visualization window (in pixel).
    visualization_window_height: str
        The height of the visualization window (in pixel).
    left_sidebar_width: str
        The width of the left side bar (in pixel).
    TODO: add more parameters.

    Attributes
    ----------
    params: dict
        The parameters to use for constructing the network.
    TODO: add more attributes
    """

    def __init__(self, theme_color="#8A2BE2", log_window_height="100px",
                 visualization_window_width="1000px",
                 visualization_window_height="500px",
                 left_sidebar_width='380px',
                 drive_widget_width="200px"):
        # set up styling.
        self.layout = {
            "theme_color": theme_color,
            "log_window_height": log_window_height,
            "visualization_window_width": visualization_window_width,
            "visualization_window_height": visualization_window_height,
            "left_sidebar_width": left_sidebar_width,
            "drive_widget_width": drive_widget_width,
        }
        self.layout['drive_widget_width'] = drive_widget_width

        # load default parameters
        self.params = self.load_parameters()

        # In-memory storage of all simulation related variables
        self.variables = dict(net=None, dpls=list())

        # Simulation parameters
        self.tstop = FloatText(value=170,
                               description='tstop (ms):',
                               disabled=False)
        self.tstep = FloatText(value=0.025,
                               description='tstep (ms):',
                               disabled=False)
        self.ntrials = IntText(value=1, description='Trials:', disabled=False)

        # select backends
        self.backend_selection = Dropdown(options=[('Joblib', 'Joblib'),
                                                   ('MPI', 'MPI')],
                                          value='Joblib',
                                          description='Backend:')

        # visualization layout
        self.viz_layout_selection = Dropdown(options=[('Horizontal', 'L-R'),
                                                      ('Vertical', 'U-D')],
                                             value='L-R',
                                             description='Layout:')

        self.mpi_cmd = Text(value='mpiexec',
                            placeholder='Fill if applies',
                            description='MPI cmd:',
                            disabled=False)

        self.n_jobs = BoundedIntText(value=1, min=1,
                                     max=multiprocessing.cpu_count(),
                                     description='Cores:', disabled=False)

        self.drive_type_selection = RadioButtons(
            options=['Evoked', 'Poisson', 'Rhythmic'],
            value='Evoked',
            description='Drive:',
            disabled=False,
            layout=Layout(width=self.layout['drive_widget_width']))

        self.location_selection = RadioButtons(
            options=['proximal', 'distal'], value='proximal',
            description='Location', disabled=False,
            layout=Layout(width=self.layout['drive_widget_width']))

        self.add_drive_button = create_expanded_button(
            'Add drive', 'primary', height='30px',
            button_color=self.layout['theme_color'])

        # Run, delete drives and load button
        self.run_button = create_expanded_button(
            'Run', 'success', height='30px',
            button_color=self.layout['theme_color'])

        self.load_button = FileUpload(
            accept='.json,.param', multiple=False,
            style={'button_color': self.layout['theme_color']},
            description='Load network',
            button_style='success')

        self.delete_drive_button = create_expanded_button(
            'Delete drives', 'success', height='30px',
            button_color=self.layout['theme_color'])

        # Visualization figure list
        self.plot_outputs_list = list()
        self.plot_dropdowns_list = list()
        # Add drive section
        self.drive_widgets = list()
        self.drive_boxes = list()

        self._init_ui_components()

    def _init_ui_components(self):
        """Initialize larger UI components and dynamical output windows. It's
        not encouraged for users to modify or access attributes in this part.
        """
        self.connectivity_sliders = self.init_cell_connectivity(self.params)
        # dynamic larger components
        self.drives_out = Output()  # window to add new drives
        self.log_out = Output(
            layout={
                'border': '1px solid gray',
                'height': self.layout['log_window_height'],
                'overflow': 'auto'
            })
        # visualization window
        self.visualization_window = Output(layout={
            'height': self.layout['visualization_window_height'],
            'width': self.layout['visualization_window_width'],
        })
        # detailed configuration of backends
        self.backend_config = Output()

        # static parts
        # Running status
        self.simulation_status = HTML(value="""<div
            style='background:gray;padding-left:10px;color:white;'>
            Not running</div>""")

        # footer
        self.footer = VBox([
            HBox([
                HBox([
                    self.run_button, self.load_button, self.delete_drive_button
                ], layout={"width": self.layout['left_sidebar_width']}),
                self.viz_layout_selection,
            ]), self.log_out, self.simulation_status
        ])
        # title
        self.header = HTML(value=f"""<div
            style='background:{self.layout['theme_color']};
            text-align:center;color:white;'>
            HUMAN NEOCORTICAL NEUROSOLVER</div>""")

    def load_parameters(self, params_fname=None):
        if not params_fname:
            # by default load default.json
            hnn_core_root = op.join(op.dirname(hnn_core.__file__))
            params_fname = op.join(hnn_core_root, 'param', 'default.json')

        return read_params(params_fname)

    @staticmethod
    def init_cell_connectivity(params):
        return [
            _get_sliders(params, [
                'gbar_L2Pyr_L2Pyr_ampa', 'gbar_L2Pyr_L2Pyr_nmda',
                'gbar_L2Basket_L2Pyr_gabaa', 'gbar_L2Basket_L2Pyr_gabab'
            ]),
            _get_sliders(params, [
                'gbar_L2Pyr_L5Pyr', 'gbar_L2Basket_L5Pyr',
                'gbar_L5Pyr_L5Pyr_ampa', 'gbar_L5Pyr_L5Pyr_nmda',
                'gbar_L5Basket_L5Pyr_gabaa', 'gbar_L5Basket_L5Pyr_gabab'
            ]),
            _get_sliders(params,
                         ['gbar_L2Pyr_L2Basket', 'gbar_L2Basket_L2Basket']),
            _get_sliders(params, [
                'gbar_L2Pyr_L5Basket', 'gbar_L5Pyr_L5Basket',
                'gbar_L5Basket_L5Basket'
            ])
        ]

    def _link_callbacks(self):
        # link callbacks
        def _handle_backend_change(backend_type):
            return handle_backend_change(backend_type.new, self.backend_config,
                                         self.mpi_cmd, self.n_jobs)

        def _add_drive_button_clicked(b):
            return add_drive_widget(self.drive_type_selection.value,
                                    self.drive_boxes, self.drive_widgets,
                                    self.drives_out, self.tstop,
                                    self.location_selection.value)

        def _delete_drives_clicked(b):
            self.drives_out.clear_output()
            # black magic: the following does not work
            # global drive_widgets; drive_widgets = list()
            while len(self.drive_widgets) > 0:
                self.drive_widgets.pop()
                self.drive_boxes.pop()

        def _on_upload_change(change):
            return on_upload_change(change, self.connectivity_sliders,
                                    self.params, self.tstop, self.tstep,
                                    self.log_out, self.variables,
                                    self.drive_boxes, self.drive_widgets,
                                    self.drives_out)

        def _run_button_clicked(b):
            return run_button_clicked(
                self.log_out, self.drive_widgets, self.variables, self.tstep,
                self.tstop, self.ntrials, self.backend_selection, self.mpi_cmd,
                self.n_jobs, self.params, self.plot_outputs_list,
                self.plot_dropdowns_list, self.simulation_status, b)

        def _handle_viz_layout_change(layout_option):
            return initialize_viz_window(
                self.visualization_window, self.variables,
                self.plot_outputs_list, self.plot_dropdowns_list,
                self.layout['visualization_window_width'],
                self.layout['visualization_window_height'],
                layout_option=layout_option.new)

        self.backend_selection.observe(_handle_backend_change, 'value')
        self.add_drive_button.on_click(_add_drive_button_clicked)
        self.delete_drive_button.on_click(_delete_drives_clicked)
        self.load_button.observe(_on_upload_change)
        self.run_button.on_click(_run_button_clicked)
        self.viz_layout_selection.observe(_handle_viz_layout_change, 'value')

    def run(self):
        # compose widgets
        simulation_box = VBox([
            self.tstop, self.tstep, self.ntrials, self.backend_selection,
            self.backend_config
        ])

        # accordians to group local-connectivity by cell type
        connectivity_boxes = [
            VBox(slider) for slider in self.connectivity_sliders]
        connectivity_names = [
            'Layer 2/3 Pyramidal', 'Layer 5 Pyramidal', 'Layer 2 Basket',
            'Layer 5 Basket'
        ]
        cell_connectivity = Accordion(children=connectivity_boxes)
        for idx, connectivity_name in enumerate(connectivity_names):
            cell_connectivity.set_title(idx, connectivity_name)

        drive_selections = VBox([
            self.drive_type_selection, self.location_selection,
            self.add_drive_button
        ])
        # from IPywidgets > 8.0
        drives_options = VBox([drive_selections, self.drives_out])
        # Tabs for left pane
        left_tab = Tab()
        left_tab.children = [simulation_box, cell_connectivity, drives_options]
        titles = ['Simulation', 'Cell connectivity', 'Drives']
        for idx, title in enumerate(titles):
            left_tab.set_title(idx, title)

        hnn_gui = AppLayout(
            header=self.header, left_sidebar=left_tab,
            right_sidebar=self.visualization_window,
            footer=self.footer,
            pane_widths=[
                self.layout['left_sidebar_width'], '0px',
                self.layout['visualization_window_width']
            ],
            pane_heights=[
                '50px', self.layout['visualization_window_height'], "1"
            ],
        )

        self._link_callbacks()

        # initialize visualization
        initialize_viz_window(self.visualization_window,
                              self.variables,
                              self.plot_outputs_list,
                              self.plot_dropdowns_list,
                              self.layout['visualization_window_width'],
                              self.layout['visualization_window_height'],
                              layout_option=self.viz_layout_selection.value,
                              init=True)

        # load initial drives
        # initialize drive ipywidgets
        load_drives(self.variables, self.params, self.log_out, self.drives_out,
                    self.drive_widgets, self.drive_boxes, self.tstop)

        return hnn_gui


def create_expanded_button(description, button_style, height, disabled=False,
                           button_color="#8A2BE2"):
    return Button(description=description, button_style=button_style,
                  layout=Layout(height=height, width='auto'),
                  style={'button_color': button_color},
                  disabled=disabled)


def _get_sliders(params, param_keys):
    """Get sliders"""
    style = {'description_width': '150px'}
    sliders = list()
    for d in param_keys:
        text_input = FloatText(value=params[d],
                               disabled=False, continuous_update=False,
                               description=d.split('gbar_')[1],
                               style=style)

        slider = FloatLogSlider(value=params[d],
                                min=-5,
                                max=1,
                                step=0.2,
                                description=" ",
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=False,
                                readout_format='.2e',
                                style=style)

        link((slider, 'value'), (text_input, 'value'))
        sliders.append(
            VBox([
                text_input, slider,
                HTML(value="<hr style='margin-bottom:5px'/>")
            ]))

    def _update_params(**updates):
        logging.debug(f'Connectivity parameters updates: {updates}')
        params.update(dict(**updates))
    # must use the first one as it could be zero.
    interactive_output(_update_params,
                       {s.children[0].description: s.children[0]
                        for s in sliders})
    return sliders


def _get_cell_specific_widgets(layout, style, location, data=None):
    default_data = {
        'weights_ampa': {
            'L5_pyramidal': 0.,
            'L2_pyramidal': 0.,
            'L5_basket': 0.,
            'L2_basket': 0.
        },
        'weights_nmda': {
            'L5_pyramidal': 0.,
            'L2_pyramidal': 0.,
            'L5_basket': 0.,
            'L2_basket': 0.
        },
        'delays': {
            'L5_pyramidal': 0.1,
            'L2_pyramidal': 0.1,
            'L5_basket': 0.1,
            'L2_basket': 0.1
        },
    }
    if isinstance(data, dict):
        for k in default_data.keys():
            if k in data and data[k] is not None:
                default_data[k].update(data[k])

    kwargs = dict(layout=layout, style=style)
    cell_types = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket', 'L2_basket']
    if location == "distal":
        cell_types.remove('L5_basket')

    weights_ampa, weights_nmda, delays = dict(), dict(), dict()
    for cell_type in cell_types:
        weights_ampa[f'{cell_type}'] = FloatText(
            value=default_data['weights_ampa'][cell_type],
            description=f'{cell_type}:',
            **kwargs)
        weights_nmda[f'{cell_type}'] = FloatText(
            value=default_data['weights_nmda'][cell_type],
            description=f'{cell_type}:',
            **kwargs)
        delays[f'{cell_type}'] = FloatText(
            value=default_data['delays'][cell_type],
            description=f'{cell_type}:',
            **kwargs)

    widgets_dict = {
        'weights_ampa': weights_ampa,
        'weights_nmda': weights_nmda,
        'delays': delays
    }
    widgets_list = ([HTML(value="<b>AMPA weights</b>")] +
                    list(weights_ampa.values()) +
                    [HTML(value="<b>NMDA weights</b>")] +
                    list(weights_nmda.values()) +
                    [HTML(value="<b>Synaptic delays</b>")] +
                    list(delays.values()))
    return widgets_list, widgets_dict


def _get_rhythmic_widget(name,
                         tstop_widget,
                         layout,
                         style,
                         location,
                         data=None,
                         default_weights_ampa=None,
                         default_weights_nmda=None,
                         default_delays=None):
    default_data = {
        'tstart': 0.,
        'tstart_std': 0.,
        'tstop': 0.,
        'burst_rate': 7.5,
        'burst_std': 0,
        'repeats': 1,
        'seedcore': 14,
    }
    if isinstance(data, dict):
        default_data.update(data)
    kwargs = dict(layout=layout, style=style)
    tstart = FloatText(value=default_data['tstart'],
                       description='Start time (ms)',
                       **kwargs)
    tstart_std = FloatText(value=default_data['tstart_std'],
                           description='Start time dev (ms)',
                           **kwargs)
    tstop = BoundedFloatText(
        value=default_data['tstop'],
        description='Stop time (ms)',
        max=tstop_widget.value,
        **kwargs,
    )
    burst_rate = FloatText(value=default_data['burst_rate'],
                           description='Burst rate (Hz)',
                           **kwargs)
    burst_std = FloatText(value=default_data['burst_std'],
                          description='Burst std dev (Hz)',
                          **kwargs)
    repeats = FloatText(value=default_data['repeats'],
                        description='Repeats',
                        **kwargs)
    seedcore = IntText(value=default_data['seedcore'],
                       description='Seed',
                       **kwargs)

    widgets_list, widgets_dict = _get_cell_specific_widgets(
        layout,
        style,
        location,
        data={
            'weights_ampa': default_weights_ampa,
            'weights_nmda': default_weights_nmda,
            'delays': default_delays,
        },
    )
    drive_box = VBox(
        [tstart, tstart_std, tstop, burst_rate, burst_std, repeats, seedcore] +
        widgets_list)
    drive = dict(type='Rhythmic',
                 name=name,
                 tstart=tstart,
                 tstart_std=tstart_std,
                 burst_rate=burst_rate,
                 burst_std=burst_std,
                 repeats=repeats,
                 seedcore=seedcore,
                 location=location,
                 tstop=tstop)
    drive.update(widgets_dict)
    return drive, drive_box


def _get_poisson_widget(name,
                        tstop_widget,
                        layout,
                        style,
                        location,
                        data=None,
                        default_weights_ampa=None,
                        default_weights_nmda=None,
                        default_delays=None):
    default_data = {
        'tstart': 0.0,
        'tstop': 0.0,
        'seedcore': 14,
        'rate_constant': {
            'L5_pyramidal': 8.5,
            'L2_pyramidal': 8.5,
            'L5_basket': 8.5,
            'L2_basket': 8.5,
        }
    }
    if isinstance(data, dict):
        default_data.update(data)
    tstart = FloatText(value=default_data['tstart'],
                       description='Start time (ms)',
                       layout=layout,
                       style=style)
    tstop = BoundedFloatText(
        value=default_data['tstop'],
        max=tstop_widget.value,
        description='Stop time (ms)',
        layout=layout,
        style=style,
    )
    seedcore = IntText(value=default_data['seedcore'],
                       description='Seed',
                       layout=layout,
                       style=style)

    cell_types = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket', 'L2_basket']
    rate_constant = dict()
    for cell_type in cell_types:
        rate_constant[f'{cell_type}'] = FloatText(
            value=default_data['rate_constant'][cell_type],
            description=f'{cell_type}:',
            layout=layout,
            style=style)

    widgets_list, widgets_dict = _get_cell_specific_widgets(
        layout,
        style,
        location,
        data={
            'weights_ampa': default_weights_ampa,
            'weights_nmda': default_weights_nmda,
            'delays': default_delays,
        },
    )
    widgets_dict.update({'rate_constant': rate_constant})
    widgets_list.extend([HTML(value="<b>Rate constants</b>")] +
                        list(widgets_dict['rate_constant'].values()))

    drive_box = VBox([tstart, tstop, seedcore] + widgets_list)
    drive = dict(
        type='Poisson',
        name=name,
        tstart=tstart,
        tstop=tstop,
        rate_constant=rate_constant,
        seedcore=seedcore,
        location=location,  # notice this is a widget but a str!
    )
    drive.update(widgets_dict)
    return drive, drive_box


def _get_evoked_widget(name,
                       layout,
                       style,
                       location,
                       data=None,
                       default_weights_ampa=None,
                       default_weights_nmda=None,
                       default_delays=None):
    default_data = {
        'mu': 0,
        'sigma': 1,
        'numspikes': 1,
        'seedcore': 14,
    }
    if isinstance(data, dict):
        default_data.update(data)
    kwargs = dict(layout=layout, style=style)
    mu = FloatText(value=default_data['mu'],
                   description='Mean time:',
                   **kwargs)
    sigma = FloatText(value=default_data['sigma'],
                      description='Std dev time:',
                      **kwargs)
    numspikes = IntText(value=default_data['numspikes'],
                        description='No. Spikes:',
                        **kwargs)
    seedcore = IntText(value=default_data['seedcore'],
                       description='Seed: ',
                       **kwargs)

    widgets_list, widgets_dict = _get_cell_specific_widgets(
        layout,
        style,
        location,
        data={
            'weights_ampa': default_weights_ampa,
            'weights_nmda': default_weights_nmda,
            'delays': default_delays,
        },
    )

    drive_box = VBox([mu, sigma, numspikes, seedcore] + widgets_list)
    drive = dict(type='Evoked',
                 name=name,
                 mu=mu,
                 sigma=sigma,
                 numspikes=numspikes,
                 seedcore=seedcore,
                 location=location,
                 sync_within_trial=False)
    drive.update(widgets_dict)
    return drive, drive_box


def add_drive_widget(drive_type,
                     drive_boxes,
                     drive_widgets,
                     drives_out,
                     tstop_widget,
                     location,
                     prespecified_drive_name=None,
                     prespecified_drive_data=None,
                     prespecified_weights_ampa=None,
                     prespecified_weights_nmda=None,
                     prespecified_delays=None,
                     render=True,
                     expand_last_drive=True,
                     event_seed=14):
    """Add a widget for a new drive."""
    layout = Layout(width='270px', height='auto')
    style = {'description_width': '150px'}
    drives_out.clear_output()
    if not prespecified_drive_data:
        prespecified_drive_data = {}
    prespecified_drive_data.update({"seedcore": max(event_seed, 2)})

    with drives_out:
        if not prespecified_drive_name:
            name = drive_type + str(len(drive_boxes))
        else:
            name = prespecified_drive_name
        logging.debug(f"add drive type to widget: {drive_type}")
        if drive_type in ('Rhythmic', 'Bursty'):
            drive, drive_box = _get_rhythmic_widget(
                name,
                tstop_widget,
                layout,
                style,
                location,
                data=prespecified_drive_data,
                default_weights_ampa=prespecified_weights_ampa,
                default_weights_nmda=prespecified_weights_nmda,
                default_delays=prespecified_delays,
            )
        elif drive_type == 'Poisson':
            drive, drive_box = _get_poisson_widget(
                name,
                tstop_widget,
                layout,
                style,
                location,
                data=prespecified_drive_data,
                default_weights_ampa=prespecified_weights_ampa,
                default_weights_nmda=prespecified_weights_nmda,
                default_delays=prespecified_delays,
            )
        elif drive_type in ('Evoked', 'Gaussian'):
            drive, drive_box = _get_evoked_widget(
                name,
                layout,
                style,
                location,
                data=prespecified_drive_data,
                default_weights_ampa=prespecified_weights_ampa,
                default_weights_nmda=prespecified_weights_nmda,
                default_delays=prespecified_delays,
            )

        if drive_type in [
                'Evoked', 'Poisson', 'Rhythmic', 'Bursty', 'Gaussian'
        ]:
            drive_boxes.append(drive_box)
            drive_widgets.append(drive)

        if render:
            accordion = Accordion(
                children=drive_boxes,
                selected_index=len(drive_boxes) -
                1 if expand_last_drive else None,
            )
            for idx, drive in enumerate(drive_widgets):
                accordion.set_title(idx,
                                    f"{drive['name']} ({drive['location']})")
            display(accordion)


def _debug_update_plot_window(variables, _plot_out, plot_type, idx):
    update_plot_window(variables, _plot_out, plot_type)


def update_plot_window(variables, _plot_out, plot_type):
    _plot_out.clear_output()
    if not (plot_type['type'] == 'change' and plot_type['name'] == 'value'):
        return

    with _plot_out:
        if plot_type['new'] == 'spikes':
            if variables['net'].cell_response:
                fig, ax = plt.subplots()
                variables['net'].cell_response.plot_spikes_raster(ax=ax)
            else:
                print("No cell response data")

        elif plot_type['new'] == 'current dipole':
            if len(variables['dpls']) > 0:
                fig, ax = plt.subplots()
                plot_dipole(variables['dpls'], ax=ax, average=True)
            else:
                print("No dipole data")

        elif plot_type['new'] == 'input histogram':
            # BUG: got error here, need a better way to handle exception
            if variables['net'].cell_response:
                fig, ax = plt.subplots()
                variables['net'].cell_response.plot_spikes_hist(ax=ax)
            else:
                print("No cell response data")

        elif plot_type['new'] == 'PSD':
            if len(variables['dpls']) > 0:
                fig, ax = plt.subplots()
                variables['dpls'][0].plot_psd(fmin=0, fmax=50, ax=ax)
            else:
                print("No dipole data")

        elif plot_type['new'] == 'spectogram':
            if len(variables['dpls']) > 0:
                freqs = np.arange(10., 100., 1.)
                n_cycles = freqs / 8.
                fig, ax = plt.subplots()
                variables['dpls'][0].plot_tfr_morlet(freqs,
                                                     n_cycles=n_cycles,
                                                     ax=ax)
            else:
                print("No dipole data")
        elif plot_type['new'] == 'network':
            if variables['net']:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                variables['net'].plot_cells(ax=ax)
            else:
                print("No network data")


def load_drives(variables, params, log_out, drives_out, drive_widgets,
                drive_boxes, tstop):
    """Add drive ipywidgets from params."""
    variables['net'] = jones_2009_model(params)
    log_out.clear_output()
    with log_out:
        drive_specs = _extract_drive_specs_from_hnn_params(
            variables['net']._params, list(variables['net'].cell_types.keys()))

        # clear before adding drives
        drives_out.clear_output()
        while len(drive_widgets) > 0:
            drive_widgets.pop()
            drive_boxes.pop()

        drive_names = sorted(drive_specs.keys())
        for idx, drive_name in enumerate(drive_names):  # order matters
            specs = drive_specs[drive_name]
            should_render = idx == (len(drive_names) - 1)

            add_drive_widget(
                specs['type'].capitalize(),
                drive_boxes,
                drive_widgets,
                drives_out,
                tstop,
                specs['location'],
                prespecified_drive_name=drive_name,
                prespecified_drive_data=specs['dynamics'],
                prespecified_weights_ampa=specs['weights_ampa'],
                prespecified_weights_nmda=specs['weights_nmda'],
                prespecified_delays=specs['synaptic_delays'],
                render=should_render,
                expand_last_drive=False,
                event_seed=specs['event_seed'],
            )


def on_upload_change(change, sliders, params, tstop, tstep, log_out, variables,
                     drive_boxes, drive_widgets, drives_out):
    if len(change['owner'].value) == 0:
        return

    # params_fname = change['owner'].metadata[0]['name']
    params_fname = list(change['owner'].value.keys())[0]
    file_uploaded = change['owner'].value
    param_data = list(file_uploaded.values())[0]['content']
    param_data = codecs.decode(param_data, encoding="utf-8")

    ext = op.splitext(params_fname)[1]
    read_func = {'.json': _read_json, '.param': _read_legacy_params}
    params_network = read_func[ext](param_data)

    log_out.clear_output()
    with log_out:
        print(f"parameter key: {params_network.keys()}")
        for slider in sliders:
            for sl in slider:
                key = 'gbar_' + sl.children[0].description
                sl.value = params_network[key]

        if 'tstop' in params_network.keys():
            tstop.value = params_network['tstop']
        if 'dt' in params_network.keys():
            tstep.value = params_network['dt']

        params.update(params_network)
    load_drives(variables, params, log_out, drives_out, drive_widgets,
                drive_boxes, tstop)


def _init_network_from_widgets(params, tstep, tstop, variables, drive_widgets):
    """Construct network and add drives."""
    print("init network")
    params['dt'] = tstep.value
    params['tstop'] = tstop.value
    variables['net'] = jones_2009_model(
        params,
        add_drives_from_params=False,
    )
    # add drives to network
    for drive in drive_widgets:
        logging.debug(f"add drive type to network: {drive['type']}")
        weights_ampa = {
            k: v.value
            for k, v in drive['weights_ampa'].items()
        }
        weights_nmda = {
            k: v.value
            for k, v in drive['weights_nmda'].items()
        }
        synaptic_delays = {k: v.value for k, v in drive['delays'].items()}
        print(
            f"drive type is {drive['type']}, location={drive['location']}")
        if drive['type'] == 'Poisson':
            rate_constant = {
                k: v.value
                for k, v in drive['rate_constant'].items() if v.value > 0
            }
            weights_ampa = {
                k: v
                for k, v in weights_ampa.items() if k in rate_constant
            }
            weights_nmda = {
                k: v
                for k, v in weights_nmda.items() if k in rate_constant
            }
            variables['net'].add_poisson_drive(
                name=drive['name'],
                tstart=drive['tstart'].value,
                tstop=drive['tstop'].value,
                rate_constant=rate_constant,
                location=drive['location'],
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                synaptic_delays=synaptic_delays,
                space_constant=100.0,
                event_seed=drive['seedcore'].value)
        elif drive['type'] in ('Evoked', 'Gaussian'):
            variables['net'].add_evoked_drive(
                name=drive['name'],
                mu=drive['mu'].value,
                sigma=drive['sigma'].value,
                numspikes=drive['numspikes'].value,
                location=drive['location'],
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                synaptic_delays=synaptic_delays,
                space_constant=3.0,
                event_seed=drive['seedcore'].value)
        elif drive['type'] in ('Rhythmic', 'Bursty'):
            variables['net'].add_bursty_drive(
                name=drive['name'],
                tstart=drive['tstart'].value,
                tstart_std=drive['tstart_std'].value,
                burst_rate=drive['burst_rate'].value,
                burst_std=drive['burst_std'].value,
                location=drive['location'],
                tstop=drive['tstop'].value,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                synaptic_delays=synaptic_delays,
                event_seed=drive['seedcore'].value)


def run_button_clicked(log_out, drive_widgets, variables, tstep, tstop,
                       ntrials, backend_selection, mpi_cmd, n_jobs,
                       params, plot_outputs_list, plot_dropdowns_list,
                       simulation_status, b):
    """Run the simulation and plot outputs."""
    log_out.clear_output()
    with log_out:
        _init_network_from_widgets(params, tstep, tstop, variables,
                                   drive_widgets)

        print("start simulation")
        if backend_selection.value == "MPI":
            variables['backend'] = MPIBackend(
                n_procs=multiprocessing.cpu_count() - 1, mpi_cmd=mpi_cmd.value)
        else:
            variables['backend'] = JoblibBackend(n_jobs=n_jobs.value)
            print(f"Using Joblib with {n_jobs.value} core(s).")
        with variables['backend']:
            simulation_status.value = """<div
            style='background:orange;padding-left:10px;color:white;'>
            Running...</div>"""
            variables['dpls'] = simulate_dipole(variables['net'],
                                                tstop=tstop.value,
                                                n_trials=ntrials.value)

            window_len, scaling_factor = 30, 3000
            simulation_status.value = """<div
            style='background:green;padding-left:10px;color:white;'>
            Simulation finished</div>"""
            for dpl in variables['dpls']:
                dpl.smooth(window_len).scale(scaling_factor)

    # Update all plotting panels
    for idx in range(len(plot_outputs_list)):
        with log_out:
            print(f"updating panel {idx}")
        update_plot_window(
            variables, plot_outputs_list[idx], {
                "type": "change",
                "name": "value",
                "new": plot_dropdowns_list[idx].value
            })


def handle_backend_change(backend_type, backend_config, mpi_cmd, n_jobs):
    backend_config.clear_output()
    with backend_config:
        if backend_type == "MPI":
            display(mpi_cmd)
        elif backend_type == "Joblib":
            display(n_jobs)


def init_left_right_viz_layout(plot_outputs,
                               plot_dropdowns,
                               window_height,
                               variables,
                               plot_options,
                               previous_outputs,
                               border='1px solid gray',
                               init=False):
    height_plot = window_height
    plot_outputs_L = Output(layout={'border': border, 'height': height_plot})

    default_plot_types = [plot_options[0], plot_options[1]]
    for idx, plot_type in enumerate(previous_outputs[:2]):
        default_plot_types[idx] = plot_type

    plot_dropdown_L = Dropdown(
        options=plot_options,
        value=default_plot_types[0],
        description='Plot:',
        disabled=False,
    )
    plot_dropdown_L.observe(
        lambda plot_type: _debug_update_plot_window(
            variables,
            plot_outputs_L,
            plot_type,
            "Left",
        ),
        'value',
    )

    plot_outputs.append(plot_outputs_L)
    plot_dropdowns.append(plot_dropdown_L)

    plot_outputs_R = Output(layout={'border': border, 'height': height_plot})

    plot_dropdown_R = Dropdown(
        options=plot_options,
        value=default_plot_types[1],
        description='Plot:',
        disabled=False,
    )
    plot_dropdown_R.observe(
        lambda plot_type: _debug_update_plot_window(
            variables,
            plot_outputs_R,
            plot_type,
            "Right",
        ),
        'value',
    )

    plot_outputs.append(plot_outputs_R)
    plot_dropdowns.append(plot_dropdown_R)

    if not init:
        update_plot_window(variables, plot_outputs_L, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[0]
        })
        update_plot_window(variables, plot_outputs_R, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[1]
        })

    grid = GridspecLayout(1, 2, height=window_height)
    grid[0, 0] = VBox([plot_dropdown_L, plot_outputs_L])
    grid[0, 1] = VBox([plot_dropdown_R, plot_outputs_R])
    return grid


def init_upper_down_viz_layout(plot_outputs,
                               plot_dropdowns,
                               window_height,
                               variables,
                               plot_options,
                               previous_outputs,
                               border='1px solid gray',
                               init=False):
    height_plot = window_height
    default_plot_types = [plot_options[0], plot_options[1]]
    for idx, plot_type in enumerate(previous_outputs[:2]):
        default_plot_types[idx] = plot_type

    plot_outputs_U = Output(layout={
        'border': border,
        'height': f"{float(height_plot[:-2])/2}px"
    })

    plot_dropdown_U = Dropdown(
        options=plot_options,
        value=default_plot_types[0],
        description='Plot:',
        disabled=False,
    )
    plot_dropdown_U.observe(
        lambda plot_type: _debug_update_plot_window(
            variables,
            plot_outputs_U,
            plot_type,
            "Left",
        ),
        'value',
    )

    plot_outputs.append(plot_outputs_U)
    plot_dropdowns.append(plot_dropdown_U)

    # Down
    plot_outputs_D = Output(layout={'border': border, 'height': height_plot})

    plot_dropdown_D = Dropdown(
        options=plot_options,
        value=default_plot_types[1],
        description='Plot:',
        disabled=False,
    )

    plot_dropdown_D.observe(
        lambda plot_type: _debug_update_plot_window(
            variables,
            plot_outputs_D,
            plot_type,
            "Right",
        ),
        'value',
    )
    plot_outputs.append(plot_outputs_D)
    plot_dropdowns.append(plot_dropdown_D)

    if not init:
        update_plot_window(variables, plot_outputs_U, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[0]
        })
        update_plot_window(variables, plot_outputs_D, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[1]
        })

    grid = GridspecLayout(2, 1, height=window_height)
    grid[0, 0] = VBox([plot_dropdown_U, plot_outputs_U])
    grid[1, 0] = VBox([plot_dropdown_D, plot_outputs_D])
    return grid


def initialize_viz_window(viz_window,
                          variables,
                          plot_outputs,
                          plot_dropdowns,
                          window_width,
                          window_height,
                          layout_option="L-R",
                          init=False):
    plot_options = [
        'current dipole', 'input histogram', 'spikes', 'PSD', 'spectogram',
        'network'
    ]
    viz_window.clear_output()
    previous_plot_outputs_values = []
    while len(plot_outputs) > 0:
        plot_outputs.pop()
        previous_plot_outputs_values.append(plot_dropdowns.pop().value)

    with viz_window:
        # Left-Rright configuration
        if layout_option == "L-R":
            grid = init_left_right_viz_layout(plot_outputs,
                                              plot_dropdowns,
                                              window_height,
                                              variables,
                                              plot_options,
                                              previous_plot_outputs_values,
                                              init=init)

        # Upper-Down configuration
        elif layout_option == "U-D":
            grid = init_upper_down_viz_layout(plot_outputs,
                                              plot_dropdowns,
                                              window_height,
                                              variables,
                                              plot_options,
                                              previous_plot_outputs_values,
                                              init=init)

        display(grid)


def launch():
    from voila.app import main
    notebook_path = op.join(op.dirname(__file__), 'hnn_widget.ipynb')
    main([notebook_path, *sys.argv[1:]])
