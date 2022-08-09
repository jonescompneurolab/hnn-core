"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>
import codecs
import io
import logging
import multiprocessing
import os.path as op
import sys
import urllib.parse
import urllib.request

import hnn_core
import matplotlib.pyplot as plt
import numpy as np
from hnn_core import (JoblibBackend, MPIBackend, jones_2009_model, read_params,
                      simulate_dipole)
from hnn_core.network import pick_connection
from hnn_core.params import (_extract_drive_specs_from_hnn_params, _read_json,
                             _read_legacy_params)
from hnn_core.viz import plot_dipole
from IPython.display import IFrame, display
from ipywidgets import (HTML, Accordion, AppLayout, BoundedFloatText,
                        BoundedIntText, Button, Dropdown, FileUpload,
                        FloatLogSlider, FloatText, GridspecLayout,
                        HBox, IntText, Layout, Output, RadioButtons, Tab, Text,
                        VBox, link)
from ipywidgets.embed import embed_minimal_html

_plot_options = {
    'Horizontal': 'L-R',
    'Vertical': 'U-D',
}


def _update_plot_window(simulation_data, _plot_out, plot_type):
    """Refresh plots with data from simulation_data."""
    _plot_out.clear_output()
    if not (plot_type['type'] == 'change' and plot_type['name'] == 'value'):
        return

    with _plot_out:
        if plot_type['new'] == 'spikes':
            if simulation_data['net'].cell_response:
                fig, ax = plt.subplots()
                simulation_data['net'].cell_response.plot_spikes_raster(ax=ax)
            else:
                print("No cell response data")

        elif plot_type['new'] == 'current dipole':
            if len(simulation_data['dpls']) > 0:
                fig, ax = plt.subplots()
                plot_dipole(simulation_data['dpls'], ax=ax, average=True)
            else:
                print("No dipole data")

        elif plot_type['new'] == 'layer-specific dipole':
            if len(simulation_data['dpls']) > 0:
                layers = ["L2", "L5", "agg"]
                fig, axes = plt.subplots(len(layers), 1, sharex=True)
                plot_dipole(simulation_data['dpls'], ax=axes,
                            layer=layers, average=True)
            else:
                print("No dipole data")

        elif plot_type['new'] == 'input histogram':
            # BUG: got error here, need a better way to handle exception
            if simulation_data['net'].cell_response:
                fig, ax = plt.subplots()
                simulation_data['net'].cell_response.plot_spikes_hist(ax=ax)
            else:
                print("No cell response data")

        elif plot_type['new'] == 'PSD':
            if len(simulation_data['dpls']) > 0:
                fig, ax = plt.subplots()
                simulation_data['dpls'][0].plot_psd(fmin=0, fmax=50, ax=ax)
            else:
                print("No dipole data")

        elif plot_type['new'] == 'spectogram':
            if len(simulation_data['dpls']) > 0:
                freqs = np.arange(10., 100., 1.)
                n_cycles = freqs / 8.
                fig, ax = plt.subplots()
                simulation_data['dpls'][0].plot_tfr_morlet(freqs,
                                                           n_cycles=n_cycles,
                                                           ax=ax)
            else:
                print("No dipole data")
        elif plot_type['new'] == 'network':
            if simulation_data['net']:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                simulation_data['net'].plot_cells(ax=ax)
            else:
                print("No network data")


def _init_left_right_viz_layout(plot_outputs, plot_dropdowns, simulation_data,
                                plot_options, previous_outputs, layout,
                                init=False):
    layout_LR = Layout(width=f"{int(layout.width[:-2])/2-10}px",
                       height=layout.height,
                       border=layout.border)

    plot_outputs_L = Output(layout=layout_LR)

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
            simulation_data,
            plot_outputs_L,
            plot_type,
            "Left",
        ),
        'value',
    )

    plot_outputs.append(plot_outputs_L)
    plot_dropdowns.append(plot_dropdown_L)

    plot_outputs_R = Output(layout=layout_LR)

    plot_dropdown_R = Dropdown(
        options=plot_options,
        value=default_plot_types[1],
        description='Plot:',
        disabled=False,
    )
    plot_dropdown_R.observe(
        lambda plot_type: _debug_update_plot_window(
            simulation_data,
            plot_outputs_R,
            plot_type,
            "Right",
        ),
        'value',
    )

    plot_outputs.append(plot_outputs_R)
    plot_dropdowns.append(plot_dropdown_R)

    if not init:
        _update_plot_window(simulation_data, plot_outputs_L, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[0]
        })
        _update_plot_window(simulation_data, plot_outputs_R, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[1]
        })
    grid = GridspecLayout(1, 2, height=layout.height)
    grid[0, 0] = VBox([plot_dropdown_L, plot_outputs_L])
    grid[0, 1] = VBox([plot_dropdown_R, plot_outputs_R])
    return grid


def _init_upper_down_viz_layout(plot_outputs, plot_dropdowns, simulation_data,
                                plot_options, previous_outputs, layout,
                                init=False):

    default_plot_types = [plot_options[0], plot_options[1]]
    for idx, plot_type in enumerate(previous_outputs[:2]):
        default_plot_types[idx] = plot_type

    plot_outputs_U = Output(layout={
        'border': layout.border,
        'height': f"{float(layout.height[:-2])/2}px"
    })

    plot_dropdown_U = Dropdown(
        options=plot_options,
        value=default_plot_types[0],
        description='Plot:',
        disabled=False,
    )
    plot_dropdown_U.observe(
        lambda plot_type: _debug_update_plot_window(
            simulation_data,
            plot_outputs_U,
            plot_type,
            "Left",
        ),
        'value',
    )

    plot_outputs.append(plot_outputs_U)
    plot_dropdowns.append(plot_dropdown_U)

    # Down
    plot_outputs_D = Output(layout={
        'border': layout.border,
        'height': layout.height
    })

    plot_dropdown_D = Dropdown(
        options=plot_options,
        value=default_plot_types[1],
        description='Plot:',
        disabled=False,
    )

    plot_dropdown_D.observe(
        lambda plot_type: _debug_update_plot_window(
            simulation_data,
            plot_outputs_D,
            plot_type,
            "Right",
        ),
        'value',
    )
    plot_outputs.append(plot_outputs_D)
    plot_dropdowns.append(plot_dropdown_D)

    if not init:
        _update_plot_window(simulation_data, plot_outputs_U, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[0]
        })
        _update_plot_window(simulation_data, plot_outputs_D, {
            "type": "change",
            "name": "value",
            "new": default_plot_types[1]
        })

    grid = GridspecLayout(2, 1, height=layout.height)
    grid[0, 0] = VBox([plot_dropdown_U, plot_outputs_U])
    grid[1, 0] = VBox([plot_dropdown_D, plot_outputs_D])
    return grid


def _change_plot_type(grid, layout, plot_idx, plot_type):
    assert layout in _plot_options.values(), ValueError("Illegal layout type")
    if layout == "L-R":
        assert 0 <= plot_idx < 2
        grid[layout].children[plot_idx].children[0].value = plot_type
        pass
    elif layout == "U-D":
        assert 0 <= plot_idx < 2
        grid[layout].children[plot_idx].children[0].value = plot_type
    else:
        pass


def _initialize_viz_window(viz_grid, viz_window, simulation_data, plot_outputs,
                           plot_dropdowns, layout, layout_option="L-R",
                           init=False):
    plot_options = [
        'current dipole',
        'layer-specific dipole',
        'input histogram',
        'spikes',
        'PSD',
        'spectogram',
        'network',
    ]
    viz_window.clear_output()
    previous_plot_outputs_values = []
    while len(plot_outputs) > 0:
        plot_outputs.pop()
        previous_plot_outputs_values.append(plot_dropdowns.pop().value)

    with viz_window:
        # Left-Rright configuration
        if layout_option == "L-R":
            grid = _init_left_right_viz_layout(plot_outputs, plot_dropdowns,
                                               simulation_data, plot_options,
                                               previous_plot_outputs_values,
                                               layout, init=init)

        # Upper-Down configuration
        elif layout_option == "U-D":
            grid = _init_upper_down_viz_layout(plot_outputs, plot_dropdowns,
                                               simulation_data, plot_options,
                                               previous_plot_outputs_values,
                                               layout, init=init)

        viz_grid[layout_option] = grid
        display(grid)


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
    drive_widget_width: str
        The width of network drive tab  (in pixel).
    header_height: str
        The height of GUI titlebar  (in pixel).
    button_height: str
        The height of buttons  (in pixel).

    Attributes
    ----------
    layout: dict
        The styling configuration of GUI.
    params: dict
        The parameters to use for constructing the network.
    simulation_data: dict
        Simulation related objects, such as net and dpls.
    widget_tstop: Widget
        Simulation stop time widget.
    widget_dt: Widget
        Simulation step size widget.
    widget_ntrials: Widget
        Widget that controls the number of trials in a single simulation.
    widget_backend_selection: Widget
        Widget that selects the backend used in simulations.
    widget_viz_layout_selection: Widget
        Widget that selects the layout of visualization window.
    widget_mpi_cmd: Widget
        Widget that specify the mpi command to use when the backend is
        MPIBackend.
    widget_n_jobs: Widget
        Widget that specify the cores in multi-trial simulations.
    widget_drive_type_selection: Widget
        Widget that is used to select the drive to be added to the network.
    widget_location_selection: Widget.
        Widget that specifies the location of network drives. Could be proximal
        or distal.
    add_drive_button: Widget
        Clickable widget that is used to add a drive to the network.
    run_button: Widget
        Clickable widget that triggers simulation.
    load_button: Widget
        Clickable widget that receives uploaded parameter files.
    delete_drive_button: Widget
        Clickable widget that clear all existing network drives.
    plot_outputs_list: list
        A list of visualization panel outputs.
    plot_dropdowns_list: list
        A list of dropdown menus that control the plot types in
        plot_outputs_list.
    drive_widgets: list
        A list of network drive widgets added by add_drive_button.
    drive_boxes: list
        A list of network drive layouts.
    connectivity_sliders: list
        A list of boxes that control the weight and probability of connections
        in the network.
    """

    def __init__(self, theme_color="#8A2BE2", log_window_height="100px",
                 visualization_window_width="1000px",
                 visualization_window_height="500px",
                 left_sidebar_width='380px',
                 drive_widget_width="200px",
                 header_height="50px",
                 button_height="30px",
                 ):
        # set up styling.

        self._total_height = int(header_height[:-2]) + int(
            visualization_window_height[:-2]) + int(button_height[:-2]) + int(
                log_window_height[:-2]) + 20
        self._total_width = int(left_sidebar_width[:-2]) + int(
            visualization_window_width[:-2])
        self.layout = {
            "header_height": header_height,
            "theme_color": theme_color,
            "button": Layout(height=button_height, width='auto'),
            "log_out": Layout(border='1px solid gray',
                              width=f"{self._total_width}px",
                              height=log_window_height,
                              overflow='auto'),
            "visualization_window": Layout(width=visualization_window_width,
                                           height=visualization_window_height,
                                           border='1px solid gray'),
            "left_sidebar": Layout(width=left_sidebar_width),
            "drive_widget": Layout(width=drive_widget_width),
            "drive_textbox": Layout(width='270px', height='auto'),
            "simulation_status_common": "background:gray;padding-left:10px",
            "simulation_status_running": "background:orange;padding-left:10px",
        }

        self._simulation_status_contents = {
            "not_running":
            f"""<div style='{self.layout['simulation_status_common']};
            color:white;'>Not running</div>""",
            "running":
            f"""<div style='{self.layout['simulation_status_running']};
            color:white;'>Running...</div>""",
            "finished":
            f"""<div style='{self.layout['simulation_status_common']};
            color:white;'>Simulation finished</div>""",
        }

        # load default parameters
        self.params = self.load_parameters()

        # In-memory storage of all simulation related simulation_data
        self.simulation_data = dict(net=None, dpls=list())

        # Simulation parameters
        self.widget_tstop = FloatText(value=170, description='tstop (ms):',
                                      disabled=False)
        self.widget_dt = FloatText(value=0.025, description='dt (ms):',
                                   disabled=False)
        self.widget_ntrials = IntText(value=1, description='Trials:',
                                      disabled=False)

        # select backends
        self.widget_backend_selection = Dropdown(options=[('Joblib', 'Joblib'),
                                                          ('MPI', 'MPI')],
                                                 value='Joblib',
                                                 description='Backend:')

        # visualization layout
        self.widget_viz_layout_selection = Dropdown(
            options=[(k, _plot_options[k]) for k in _plot_options],
            value=list(_plot_options.values())[0],
            description='Layout:')

        self.widget_mpi_cmd = Text(value='mpiexec',
                                   placeholder='Fill if applies',
                                   description='MPI cmd:', disabled=False)

        self.widget_n_jobs = BoundedIntText(value=1, min=1,
                                            max=multiprocessing.cpu_count(),
                                            description='Cores:',
                                            disabled=False)

        self.widget_drive_type_selection = RadioButtons(
            options=['Evoked', 'Poisson', 'Rhythmic'],
            value='Evoked',
            description='Drive:',
            disabled=False,
            layout=self.layout['drive_widget'])

        self.widget_location_selection = RadioButtons(
            options=['proximal', 'distal'], value='proximal',
            description='Location', disabled=False,
            layout=self.layout['drive_widget'])

        self.add_drive_button = create_expanded_button(
            'Add drive', 'primary', layout=self.layout['button'],
            button_color=self.layout['theme_color'])

        # Run, delete drives and load button
        self.run_button = create_expanded_button(
            'Run', 'success', layout=self.layout['button'],
            button_color=self.layout['theme_color'])

        self.load_button = FileUpload(
            accept='.json,.param', multiple=False,
            style={'button_color': self.layout['theme_color']},
            description='Load network', layout=self.layout['button'],
            button_style='success')

        self.clear_button = create_expanded_button(
            'Clear uploaded parameters', 'danger',
            layout=self.layout['button'],
            button_color=self.layout['theme_color'])

        self.delete_drive_button = create_expanded_button(
            'Delete drives', 'success', layout=self.layout['button'],
            button_color=self.layout['theme_color'])

        # Visualization figure list
        self.plot_outputs_list = list()
        self.plot_dropdowns_list = list()

        # Add drive section
        self.drive_widgets = list()
        self.drive_boxes = list()

        # Connectivity list
        self.connectivity_widgets = list()

        self._init_ui_components()

    def _init_ui_components(self):
        """Initialize larger UI components and dynamical output windows.

        It's not encouraged for users to modify or access attributes in this
        part.
        """
        # Reloading status.
        self._load_info = {"count": 0, "prev_param_data": b""}

        # dynamic larger components
        self._drives_out = Output()  # tab to add new drives
        self._connectivity_out = Output()  # tab to tune connectivity.
        self._log_out = Output()
        # visualization window
        self._visualization_window = Output(
            layout=self.layout['visualization_window'])
        # detailed configuration of backends
        self._backend_config_out = Output()

        # static parts
        # Running status
        self._simulation_status_bar = HTML(
            value=self._simulation_status_contents['not_running'])

        # footer
        self._footer = VBox([
            HBox([
                HBox([
                    self.run_button, self.load_button, self.clear_button,
                    self.delete_drive_button
                ]),
                self.widget_viz_layout_selection,
            ]),
            HBox([self._log_out], layout=self.layout['log_out']),
            self._simulation_status_bar
        ])
        # title
        self._header = HTML(value=f"""<div
            style='background:{self.layout['theme_color']};
            text-align:center;color:white;'>
            HUMAN NEOCORTICAL NEUROSOLVER</div>""")

        # visualiation components
        self._viz_grid = {
            "L-R": None,
            "U-D": None,
        }

    def load_parameters(self, params_fname=None):
        """Read parameters from file."""
        if not params_fname:
            # by default load default.json
            hnn_core_root = op.join(op.dirname(hnn_core.__file__))
            params_fname = op.join(hnn_core_root, 'param', 'default.json')

        return read_params(params_fname)

    def _link_callbacks(self):
        """Link callbacks to UI components."""
        def _handle_backend_change(backend_type):
            return handle_backend_change(backend_type.new,
                                         self._backend_config_out,
                                         self.widget_mpi_cmd,
                                         self.widget_n_jobs)

        def _add_drive_button_clicked(b):
            return add_drive_widget(self.widget_drive_type_selection.value,
                                    self.drive_boxes, self.drive_widgets,
                                    self._drives_out, self.widget_tstop,
                                    self.widget_location_selection.value,
                                    layout=self.layout['drive_textbox'])

        def _delete_drives_clicked(b):
            self._drives_out.clear_output()
            # black magic: the following does not work
            # global drive_widgets; drive_widgets = list()
            while len(self.drive_widgets) > 0:
                self.drive_widgets.pop()
                self.drive_boxes.pop()

        def _on_upload_change(change):
            with self._log_out:
                print("received new uploaded params file...")
            return on_upload_change(change, self.params, self.widget_tstop,
                                    self.widget_dt, self._log_out,
                                    self.simulation_data, self.drive_boxes,
                                    self.drive_widgets, self._drives_out,
                                    self._connectivity_out,
                                    self.connectivity_widgets, self._load_info,
                                    layout=self.layout['drive_textbox'])

        def _run_button_clicked(b):
            return run_button_clicked(
                self._log_out, self.drive_widgets, self.simulation_data,
                self.widget_dt, self.widget_tstop, self.widget_ntrials,
                self.widget_backend_selection, self.widget_mpi_cmd,
                self.widget_n_jobs, self.params, self.plot_outputs_list,
                self.plot_dropdowns_list, self._simulation_status_bar,
                self._simulation_status_contents,
                self.connectivity_widgets, b)

        def _handle_viz_layout_change(layout_option):
            return _initialize_viz_window(
                self._viz_grid,
                self._visualization_window, self.simulation_data,
                self.plot_outputs_list, self.plot_dropdowns_list,
                self.layout['visualization_window'],
                layout_option=layout_option.new)

        def _clear_params(b):
            self._load_info["count"] = 0
            self._load_info["prev_param_data"] = 0

        self.widget_backend_selection.observe(_handle_backend_change, 'value')
        self.add_drive_button.on_click(_add_drive_button_clicked)
        self.delete_drive_button.on_click(_delete_drives_clicked)
        self.load_button.observe(_on_upload_change)
        self.run_button.on_click(_run_button_clicked)
        self.clear_button.on_click(_clear_params)
        self.widget_viz_layout_selection.observe(_handle_viz_layout_change,
                                                 'value')

    def compose(self, return_layout=True):
        """Compose widgets.

        Parameters
        ----------
        return_layout: bool
            If the method returns the layout object which can be rendered by
            IPython.display.display() method.
        """

        simulation_box = VBox([
            self.widget_tstop, self.widget_dt, self.widget_ntrials,
            self.widget_backend_selection, self._backend_config_out
        ])

        # accordians to group local-connectivity by cell type
        connectivity_boxes = [
            VBox(slider) for slider in self.connectivity_widgets]
        connectivity_names = [
            'Layer 2/3 Pyramidal', 'Layer 5 Pyramidal', 'Layer 2 Basket',
            'Layer 5 Basket'
        ]
        cell_connectivity = Accordion(children=connectivity_boxes)
        for idx, connectivity_name in enumerate(connectivity_names):
            cell_connectivity.set_title(idx, connectivity_name)

        drive_selections = VBox([
            self.widget_drive_type_selection, self.widget_location_selection,
            self.add_drive_button
        ])
        # from IPywidgets > 8.0
        drives_options = VBox([drive_selections, self._drives_out])
        # Tabs for left pane
        left_tab = Tab()
        left_tab.children = [
            simulation_box, self._connectivity_out, drives_options
        ]
        titles = ['Simulation', 'Cell connectivity', 'Drives']
        for idx, title in enumerate(titles):
            left_tab.set_title(idx, title)

        self.app_layout = AppLayout(
            header=self._header,
            left_sidebar=left_tab,
            right_sidebar=self._visualization_window,
            footer=self._footer,
            pane_widths=[
                self.layout['left_sidebar'].width, '0px',
                self.layout['visualization_window'].width
            ],
            pane_heights=[
                self.layout['header_height'],
                self.layout['visualization_window'].height, "1"
            ],
        )

        self._link_callbacks()

        # initialize visualization
        _initialize_viz_window(
            self._viz_grid,
            self._visualization_window,
            self.simulation_data,
            self.plot_outputs_list,
            self.plot_dropdowns_list,
            self.layout['visualization_window'],
            layout_option=self.widget_viz_layout_selection.value,
            init=True)

        # initialize drive and connectivity ipywidgets
        load_drive_and_connectivity(self.simulation_data, self.params,
                                    self._log_out, self._drives_out,
                                    self.drive_widgets, self.drive_boxes,
                                    self._connectivity_out,
                                    self.connectivity_widgets,
                                    self.widget_tstop, self._load_info,
                                    self.layout)

        if not return_layout:
            return
        else:
            return self.app_layout

    def show(self):
        display(self.app_layout)

    def capture(self, width=None, height=None, render=True):
        """Take a screenshot of the current GUI.

        Parameters
        ----------
        width : int | None
            The width of iframe window use to show the snapshot.
        height : int | None
            The height of iframe window use to show the snapshot.
        render: bool
            Will return an IFrame object if False

        Returns
        -------
        snapshot : An iframe snapshot object that can be rendered in notebooks.
        """
        file = io.StringIO()
        embed_minimal_html(file, views=[self.app_layout], title='')
        if not width:
            width = self._total_width + 20
        if not height:
            height = self._total_height + 20

        content = urllib.parse.quote(file.getvalue().encode('utf8'))
        data_url = f"data:text/html,{content}"
        screenshot = IFrame(data_url, width=width, height=height)
        if render:
            display(screenshot)
        else:
            return screenshot

    def run_notebook_cells(self):
        """Run all but the last cells sequentially in a Jupyter notebook.

        To properly use this function:
            1. Put this into the penultimate cell.
            2. init the HNNGUI in a single cell.
            3. Hit 'run all' button to run the whole notebook and it will
               selectively run twice.
        """
        js_string = """
        function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
        }
        function getRunningStatus(idx){
            const htmlContent = Jupyter.notebook.get_cell(idx).element[0];
            return htmlContent.childNodes[0].childNodes[0].textContent;
        }
        function cellContainsInitOrMarkdown(idx){
            const cell = Jupyter.notebook.get_cell(idx);
            if(cell.cell_type!=='code'){
                return true;
            }
            else{
                const textVal = cell.element[0].childNodes[0].textContent;
                return textVal.includes('HNNGUI()') || textVal.includes(
                    'HNNGUI');
            }
        }
        function cellContainsRunCells(idx){
            const textVal = Jupyter.notebook.get_cell(
                idx).element[0].childNodes[0].textContent;
            return textVal.includes('run_notebook_cells()');
        }
        async function runNotebook() {
            console.log("run notebook cell by cell");
            const cellHtmlContents = Jupyter.notebook.element[0].children[0];
            const nCells = cellHtmlContents.childElementCount;
            console.log(`In total we have ${nCells} cells`);

            for(let i=1; i<nCells-1; i++){
                if(cellContainsRunCells(i)){
                    break
                }
                else if(cellContainsInitOrMarkdown(i)){
                    console.log(`Skip init or markdown cell ${i}...`);
                    continue
                }
                else{
                    console.log(`About to execute cell ${i}..`);
                    Jupyter.notebook.execute_cells([i]);
                    while (getRunningStatus(i).includes("*")){
                        console.log("Still running, wait for another 2 secs");
                        await sleep(2000);
                    }
                    await sleep(1000);
                }
            }
            console.log('Done');
        }
        runNotebook();
        """
        return js_string

    # below are a series of methods that are used to manipulate the GUI
    def _simulate_upload_file(self, file_url):
        params_name = file_url.split("/")[-1]
        data = urllib.request.urlopen(file_url)
        content = b""
        for line in data:
            content += line

        uploaded_value = {
            params_name: {
                'metadata': {
                    'name': params_name,
                    'type': 'application/json',
                    'size': len(content),
                },
                'content': content
            }
        }
        self.load_button.set_trait('value', uploaded_value)

    def _simulate_left_tab_click(self, tab_title):
        tab_index = None
        for idx in self.app_layout.left_sidebar._titles.keys():
            if tab_title == self.app_layout.left_sidebar._titles[idx]:
                tab_index = int(idx)
                break
        if tab_index is None:
            raise ValueError("Incorrect tab title")
        self.app_layout.left_sidebar.selected_index = tab_index

    def _simulate_switch_plot_type(self, plot_idx, plot_type):
        _change_plot_type(self._viz_grid,
                          self.widget_viz_layout_selection.value, plot_idx,
                          plot_type)


def create_expanded_button(description, button_style, layout, disabled=False,
                           button_color="#8A2BE2"):
    return Button(description=description, button_style=button_style,
                  layout=layout, style={'button_color': button_color},
                  disabled=disabled)


def _get_connectivity_widgets(conn_data):
    """Create connectivity box widgets from specified weight and probability"""

    style = {'description_width': '150px'}
    style = {}
    sliders = list()
    for receptor_name in conn_data.keys():
        w_text_input = FloatText(value=conn_data[receptor_name]['weight'],
                                 disabled=False, continuous_update=False,
                                 description="weight",
                                 style=style)

        w_slider = FloatLogSlider(value=conn_data[receptor_name]['weight'],
                                  min=-5, max=1, step=0.2,
                                  description=" ",
                                  disabled=False,
                                  continuous_update=False,
                                  orientation='horizontal',
                                  readout=False,
                                  readout_format='.2e',
                                  style=style)

        link((w_slider, 'value'), (w_text_input, 'value'))
        conn_widget = VBox([
            HTML(value=f"""<p>
            Receptor: {conn_data[receptor_name]['receptor']}</p>"""),
            w_text_input, w_slider,
            HTML(value="<hr style='margin-bottom:5px'/>")
        ])

        conn_widget._belongsto = {
            "receptor": conn_data[receptor_name]['receptor'],
            "location": conn_data[receptor_name]['location'],
            "src_gids": conn_data[receptor_name]['src_gids'],
            "target_gids": conn_data[receptor_name]['target_gids'],
        }
        sliders.append(conn_widget)

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


def _get_rhythmic_widget(name, tstop_widget, layout, style, location,
                         data=None, default_weights_ampa=None,
                         default_weights_nmda=None, default_delays=None):
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


def _get_poisson_widget(name, tstop_widget, layout, style, location, data=None,
                        default_weights_ampa=None, default_weights_nmda=None,
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


def _get_evoked_widget(name, layout, style, location, data=None,
                       default_weights_ampa=None, default_weights_nmda=None,
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


def add_drive_widget(drive_type, drive_boxes, drive_widgets, drives_out,
                     tstop_widget, location, layout,
                     prespecified_drive_name=None,
                     prespecified_drive_data=None,
                     prespecified_weights_ampa=None,
                     prespecified_weights_nmda=None,
                     prespecified_delays=None, render=True,
                     expand_last_drive=True, event_seed=14):
    """Add a widget for a new drive."""

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


def _debug_update_plot_window(simulation_data, _plot_out, plot_type, idx):
    _update_plot_window(simulation_data, _plot_out, plot_type)


def add_connectivity_tab(simulation_data, connectivity_out,
                         connectivity_sliders):
    """Add all possible connectivity boxes to connectivity tab."""
    cell_types = [ct for ct in simulation_data['net'].cell_types.keys()]
    receptors = ('ampa', 'nmda', 'gabaa', 'gabab')
    locations = ('proximal', 'distal', 'soma')

    # clear existing connectivity
    connectivity_out.clear_output()
    while len(connectivity_sliders) > 0:
        connectivity_sliders.pop()

    connectivity_names = []
    for src_gids in cell_types:
        for target_gids in cell_types:
            for location in locations:
                # the connectivity list should be built on this level
                receptor_related_conn = {}
                for receptor in receptors:
                    conn_indices = pick_connection(net=simulation_data['net'],
                                                   src_gids=src_gids,
                                                   target_gids=target_gids,
                                                   loc=location,
                                                   receptor=receptor)
                    if len(conn_indices) > 0:
                        assert len(conn_indices) == 1
                        conn_idx = conn_indices[0]
                        current_w = simulation_data['net'].connectivity[
                            conn_idx]['nc_dict']['A_weight']
                        current_p = simulation_data['net'].connectivity[
                            conn_idx]['probability']
                        # valid connection
                        receptor_related_conn[receptor] = {
                            "weight": current_w,
                            "probability": current_p,
                            # info used to identify connection
                            "receptor": receptor,
                            "location": location,
                            "src_gids": src_gids,
                            "target_gids": target_gids,
                        }
                if len(receptor_related_conn) > 0:
                    connectivity_names.append(
                        f"{src_gids}→{target_gids} ({location})")
                    connectivity_sliders.append(
                        _get_connectivity_widgets(receptor_related_conn))

    connectivity_boxes = [VBox(slider) for slider in connectivity_sliders]
    cell_connectivity = Accordion(children=connectivity_boxes)
    for idx, connectivity_name in enumerate(connectivity_names):
        cell_connectivity.set_title(idx, connectivity_name)

    with connectivity_out:
        display(cell_connectivity)


def load_drive_and_connectivity(simulation_data, params, log_out, drives_out,
                                drive_widgets, drive_boxes, connectivity_out,
                                connectivity_sliders, tstop, load_info,
                                layout):
    """Add drive and connectivity ipywidgets from params."""
    load_info['count'] += 1
    # init the network.
    simulation_data['net'] = jones_2009_model(params)

    # Add connectivity
    add_connectivity_tab(simulation_data, connectivity_out,
                         connectivity_sliders)

    # Add drives
    log_out.clear_output()
    with log_out:
        drive_specs = _extract_drive_specs_from_hnn_params(
            simulation_data['net']._params,
            list(simulation_data['net'].cell_types.keys()))

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
                layout=layout,
                prespecified_drive_name=drive_name,
                prespecified_drive_data=specs['dynamics'],
                prespecified_weights_ampa=specs['weights_ampa'],
                prespecified_weights_nmda=specs['weights_nmda'],
                prespecified_delays=specs['synaptic_delays'],
                render=should_render,
                expand_last_drive=False,
                event_seed=specs['event_seed'],
            )


def on_upload_change(change, params, tstop, dt, log_out, simulation_data,
                     drive_boxes, drive_widgets, drives_out, connectivity_out,
                     connectivity_sliders, load_info, layout):
    if len(change['owner'].value) == 0:
        return

    # params_fname = change['owner'].metadata[0]['name']
    params_fname = list(change['owner'].value.keys())[0]
    file_uploaded = change['owner'].value
    param_data = list(file_uploaded.values())[0]['content']
    param_data = codecs.decode(param_data, encoding="utf-8")

    if load_info['prev_param_data'] == param_data:
        with log_out:
            print(
                "Same param. No reloading."
                "To force reloading, hit \"clear uploaded parameters\" button",
            )
        return
    else:
        load_info['prev_param_data'] = param_data

    ext = op.splitext(params_fname)[1]
    read_func = {'.json': _read_json, '.param': _read_legacy_params}
    params_network = read_func[ext](param_data)

    # update simulation settings and params
    log_out.clear_output()
    with log_out:
        if 'tstop' in params_network.keys():
            tstop.value = params_network['tstop']
        if 'dt' in params_network.keys():
            dt.value = params_network['dt']

        params.update(params_network)
    # init network, add drives & connectivity
    load_drive_and_connectivity(simulation_data, params, log_out, drives_out,
                                drive_widgets, drive_boxes, connectivity_out,
                                connectivity_sliders, tstop, load_info,
                                layout)


def _init_network_from_widgets(params, dt, tstop, simulation_data,
                               drive_widgets, connectivity_sliders,
                               add_drive=True):
    """Construct network and add drives."""
    print("init network")
    params['dt'] = dt.value
    params['tstop'] = tstop.value
    simulation_data['net'] = jones_2009_model(
        params,
        add_drives_from_params=False,
    )
    # adjust connectivity according to the connectivity_tab
    for connectivity_slider in connectivity_sliders:
        for vbox in connectivity_slider:
            conn_indices = pick_connection(
                net=simulation_data['net'],
                src_gids=vbox._belongsto['src_gids'],
                target_gids=vbox._belongsto['target_gids'],
                loc=vbox._belongsto['location'],
                receptor=vbox._belongsto['receptor'])

            if len(conn_indices) > 0:
                assert len(conn_indices) == 1
                conn_idx = conn_indices[0]
                simulation_data['net'].connectivity[conn_idx]['nc_dict'][
                    'A_weight'] = vbox.children[1].value
                simulation_data['net'].connectivity[conn_idx][
                    'probability'] = vbox.children[3].value

    if add_drive is False:
        return
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
            simulation_data['net'].add_poisson_drive(
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
            simulation_data['net'].add_evoked_drive(
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
            simulation_data['net'].add_bursty_drive(
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


def run_button_clicked(log_out, drive_widgets, simulation_data, dt, tstop,
                       ntrials, backend_selection, mpi_cmd, n_jobs,
                       params, plot_outputs_list, plot_dropdowns_list,
                       simulation_status_bar, simulation_status_contents,
                       connectivity_sliders, b):
    """Run the simulation and plot outputs."""
    log_out.clear_output()
    with log_out:
        _init_network_from_widgets(params, dt, tstop, simulation_data,
                                   drive_widgets, connectivity_sliders)

        print("start simulation")
        if backend_selection.value == "MPI":
            simulation_data['backend'] = MPIBackend(
                n_procs=multiprocessing.cpu_count() - 1, mpi_cmd=mpi_cmd.value)
        else:
            simulation_data['backend'] = JoblibBackend(n_jobs=n_jobs.value)
            print(f"Using Joblib with {n_jobs.value} core(s).")
        with simulation_data['backend']:
            simulation_status_bar.value = simulation_status_contents['running']
            simulation_data['dpls'] = simulate_dipole(simulation_data['net'],
                                                      tstop=tstop.value,
                                                      dt=dt.value,
                                                      n_trials=ntrials.value)

            window_len, scaling_factor = 30, 3000
            simulation_status_bar.value = simulation_status_contents[
                'finished']
            for dpl in simulation_data['dpls']:
                dpl.smooth(window_len).scale(scaling_factor)

    # Update all plotting panels
    for idx in range(len(plot_outputs_list)):
        with log_out:
            print(f"updating panel {idx}")
        _update_plot_window(
            simulation_data, plot_outputs_list[idx], {
                "type": "change",
                "name": "value",
                "new": plot_dropdowns_list[idx].value
            })


def handle_backend_change(backend_type, backend_config, mpi_cmd, n_jobs):
    """Switch backends between MPI and Joblib."""
    backend_config.clear_output()
    with backend_config:
        if backend_type == "MPI":
            display(mpi_cmd)
        elif backend_type == "Joblib":
            display(n_jobs)


def launch():
    """Launch voila with hnn_widget.ipynb.

    You can pass voila commandline parameters as usual.
    """
    from voila.app import main
    notebook_path = op.join(op.dirname(__file__), 'hnn_widget.ipynb')
    main([notebook_path, *sys.argv[1:]])
