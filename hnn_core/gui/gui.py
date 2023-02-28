"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>
import codecs
import io
import logging
import multiprocessing
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from IPython.display import IFrame, display
from ipywidgets import (HTML, Accordion, AppLayout, BoundedFloatText,
                        BoundedIntText, Button, Dropdown, FileUpload,
                        FloatText, HBox, IntText, Layout, Output, RadioButtons,
                        Tab, Text, VBox)
from ipywidgets.embed import embed_minimal_html

import hnn_core
from hnn_core import (JoblibBackend, MPIBackend, jones_2009_model, read_params,
                      simulate_dipole)
from hnn_core.gui._logging import logger
from hnn_core.gui._viz_manager import _VizManager, _idx2figname
from hnn_core.network import pick_connection
from hnn_core.params import (_extract_drive_specs_from_hnn_params, _read_json,
                             _read_legacy_params)


class _OutputWidgetHandler(logging.Handler):
    def __init__(self, output_widget, *args, **kwargs):
        super(_OutputWidgetHandler, self).__init__(*args, **kwargs)
        self.out = output_widget

    def emit(self, record):
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs


class HNNGUI:
    """HNN GUI class

    Parameters
    ----------
    theme_color : str
        The theme color of the whole dashboard.
    total_height : int
        The height of the GUI (in pixel, same for all following parameters).
    total_width : int
        The width of the GUI.
    header_height : int
        The height of the header.
    button_height : int
        The height of buttons.
    operation_box_height : int
        The operation_box_height of operations box.
    drive_widget_width : int
        The width of GUI drive box.
    left_sidebar_width : int
        The width of left sidebad.
    log_window_height : int
        The height of logging window.
    status_height : int
        The height of status bar.
    dpi : int
        The screen dpi.

    Attributes
    ----------
    layout : dict
        The styling configuration of GUI.
    params : dict
        The parameters to use for constructing the network.
    simulation_data : dict
        Simulation related objects, such as net and dpls.
    widget_tstop : Widget
        Simulation stop time widget.
    widget_dt : Widget
        Simulation step size widget.
    widget_ntrials : Widget
        Widget that controls the number of trials in a single simulation.
    widget_backend_selection : Widget
        Widget that selects the backend used in simulations.
    widget_viz_layout_selection : Widget
        Widget that selects the layout of visualization window.
    widget_mpi_cmd : Widget
        Widget that specify the mpi command to use when the backend is
        MPIBackend.
    widget_n_jobs : Widget
        Widget that specify the cores in multi-trial simulations.
    widget_drive_type_selection : Widget
        Widget that is used to select the drive to be added to the network.
    widget_location_selection : Widget.
        Widget that specifies the location of network drives. Could be proximal
        or distal.
    add_drive_button : Widget
        Clickable widget that is used to add a drive to the network.
    run_button : Widget
        Clickable widget that triggers simulation.
    load_button : Widget
        Clickable widget that receives uploaded parameter files.
    delete_drive_button : Widget
        Clickable widget that clear all existing network drives.
    plot_outputs_dict : list
        A list of visualization panel outputs.
    plot_dropdown_types_dict : list
        A list of dropdown menus that control the plot types in
        plot_outputs_dict.
    drive_widgets : list
        A list of network drive widgets added by add_drive_button.
    drive_boxes : list
        A list of network drive layouts.
    connectivity_textfields : list
        A list of boxes that control the weight and probability of connections
        in the network.
    """

    def __init__(self, theme_color="#8A2BE2",
                 total_height=800,
                 total_width=1300,
                 header_height=50,
                 button_height=30,
                 operation_box_height=60,
                 drive_widget_width=200,
                 left_sidebar_width=576,
                 log_window_height=150,
                 status_height=30,
                 dpi=96,
                 ):
        # set up styling.
        self.total_height = total_height
        self.total_width = total_width

        viz_win_width = self.total_width - left_sidebar_width
        main_content_height = self.total_height - status_height

        config_box_height = main_content_height - (log_window_height +
                                                   operation_box_height)
        self.layout = {
            "dpi": dpi,
            "header_height": f"{header_height}px",
            "theme_color": theme_color,
            "btn": Layout(height=f"{button_height}px", width='auto'),
            "btn_full_w": Layout(height=f"{button_height}px", width='100%'),
            "del_fig_btn": Layout(height=f"{button_height}px", width='auto'),
            "log_out": Layout(border='1px solid gray',
                              height=f"{log_window_height-10}px",
                              overflow='auto'),
            "viz_config": Layout(width='99%'),
            "visualization_window": Layout(
                width=f"{viz_win_width-10}px",
                height=f"{main_content_height-10}px",
                border='1px solid gray',
                overflow='scroll'),
            "visualization_output": Layout(
                width=f"{viz_win_width-50}px",
                height=f"{main_content_height-100}px",
                border='1px solid gray',
                overflow='scroll'),
            "left_sidebar": Layout(width=f"{left_sidebar_width}px",
                                   height=f"{main_content_height}px"),
            "left_tab": Layout(width=f"{left_sidebar_width}px",
                               height=f"{config_box_height}px"),
            "operation_box": Layout(width=f"{left_sidebar_width}px",
                                    height=f"{operation_box_height}px",
                                    flex_wrap="wrap",
                                    ),
            "config_box": Layout(width=f"{left_sidebar_width}px",
                                 height=f"{config_box_height-100}px"),
            "drive_widget": Layout(width="auto"),
            "drive_textbox": Layout(width='270px', height='auto'),
            # simulation status related
            "simulation_status_height": f"{status_height}px",
            "simulation_status_common": "background:gray;padding-left:10px",
            "simulation_status_running": "background:orange;padding-left:10px",
            "simulation_status_failed": "background:red;padding-left:10px",
            "simulation_status_finished": "background:green;padding-left:10px",
        }

        self._simulation_status_contents = {
            "not_running":
            f"""<div style='{self.layout['simulation_status_common']};
            color:white;'>Not running</div>""",
            "running":
            f"""<div style='{self.layout['simulation_status_running']};
            color:white;'>Running...</div>""",
            "finished":
            f"""<div style='{self.layout['simulation_status_finished']};
            color:white;'>Simulation finished</div>""",
            "failed":
            f"""<div style='{self.layout['simulation_status_failed']};
            color:white;'>Simulation failed</div>""",
        }

        # load default parameters
        self.params = self.load_parameters()

        # In-memory storage of all simulation and visualization related data
        self.simulation_data = defaultdict(lambda: dict(net=None, dpls=list()))

        # Simulation parameters
        self.widget_tstop = FloatText(value=170, description='tstop (ms):',
                                      disabled=False)
        self.widget_dt = FloatText(value=0.025, description='dt (ms):',
                                   disabled=False)
        self.widget_ntrials = IntText(value=1, description='Trials:',
                                      disabled=False)
        self.widget_simulation_name = Text(value='default',
                                           placeholder='ID of your simulation',
                                           description='Name:',
                                           disabled=False)
        self.widget_backend_selection = Dropdown(options=[('Joblib', 'Joblib'),
                                                          ('MPI', 'MPI')],
                                                 value='Joblib',
                                                 description='Backend:')
        self.widget_mpi_cmd = Text(value='mpiexec',
                                   placeholder='Fill if applies',
                                   description='MPI cmd:', disabled=False)
        self.widget_n_jobs = BoundedIntText(value=1, min=1,
                                            max=multiprocessing.cpu_count(),
                                            description='Cores:',
                                            disabled=False)

        # Drive selection
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
            'Add drive', 'primary', layout=self.layout['btn'],
            button_color=self.layout['theme_color'])

        # Dashboard level buttons
        self.run_button = create_expanded_button(
            'Run', 'success', layout=self.layout['btn'],
            button_color=self.layout['theme_color'])

        self.load_connectivity_button = FileUpload(
            accept='.json,.param', multiple=False,
            style={'button_color': self.layout['theme_color']},
            description='Load local network connectivity',
            layout=self.layout['btn_full_w'], button_style='success')
        self.load_drives_button = FileUpload(
            accept='.json,.param', multiple=False,
            style={'button_color': self.layout['theme_color']},
            description='Load external drives', layout=self.layout['btn'],
            button_style='success')

        self.delete_drive_button = create_expanded_button(
            'Delete drives', 'success', layout=self.layout['btn'],
            button_color=self.layout['theme_color'])

        # Plotting window

        # Visualization figure related dicts
        self.plot_outputs_dict = dict()
        self.plot_dropdown_types_dict = dict()
        self.plot_sim_selections_dict = dict()

        # Add drive section
        self.drive_widgets = list()
        self.drive_boxes = list()

        # Connectivity list
        self.connectivity_widgets = list()

        self._init_ui_components()
        self.add_logging_window_logger()

    def add_logging_window_logger(self):
        handler = _OutputWidgetHandler(self._log_out)
        handler.setFormatter(
            logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    def _init_ui_components(self):
        """Initialize larger UI components and dynamical output windows.

        It's not encouraged for users to modify or access attributes in this
        part.
        """
        # dynamic larger components
        self._drives_out = Output()  # tab to add new drives
        self._connectivity_out = Output()  # tab to tune connectivity.

        self._log_out = Output()

        self.viz_manager = _VizManager(self.data, self.layout)

        # detailed configuration of backends
        self._backend_config_out = Output()

        # static parts
        # Running status
        self._simulation_status_bar = HTML(
            value=self._simulation_status_contents['not_running'])

        self._log_window = HBox([self._log_out], layout=self.layout['log_out'])
        self._operation_buttons = HBox([self.run_button],
                                       layout=self.layout['operation_box'])
        # title
        self._header = HTML(value=f"""<div
            style='background:{self.layout['theme_color']};
            text-align:center;color:white;'>
            HUMAN NEOCORTICAL NEUROSOLVER</div>""")

    @property
    def analysis_config(self):
        """Provides everything viz window needs except for the data."""
        return {
            "viz_style": self.layout['visualization_output'],
            # widgets
            "plot_outputs": self.plot_outputs_dict,
            "plot_dropdowns": self.plot_dropdown_types_dict,
            "plot_sim_selections": self.plot_sim_selections_dict,
            "current_sim_name": self.widget_simulation_name.value,
        }

    @property
    def data(self):
        """Provides easy access to simulation-related data."""
        return {"simulation_data": self.simulation_data}

    @staticmethod
    def load_parameters(params_fname=None):
        """Read parameters from file."""
        if not params_fname:
            # by default load default.json
            hnn_core_root = Path(hnn_core.__file__).parent
            params_fname = hnn_core_root / 'param/default.json'
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

        def _on_upload_connectivity(change):
            return on_upload_change(change, self.params, self.widget_tstop,
                                    self.widget_dt, self._log_out,
                                    self.drive_boxes, self.drive_widgets,
                                    self._drives_out, self._connectivity_out,
                                    self.connectivity_widgets,
                                    self.layout['drive_textbox'],
                                    "connectivity")

        def _on_upload_drives(change):
            return on_upload_change(change, self.params, self.widget_tstop,
                                    self.widget_dt, self._log_out,
                                    self.drive_boxes, self.drive_widgets,
                                    self._drives_out, self._connectivity_out,
                                    self.connectivity_widgets,
                                    self.layout['drive_textbox'],
                                    "drives")

        def _run_button_clicked(b):
            return run_button_clicked(
                self.widget_simulation_name, self._log_out, self.drive_widgets,
                self.data, self.widget_dt, self.widget_tstop,
                self.widget_ntrials, self.widget_backend_selection,
                self.widget_mpi_cmd, self.widget_n_jobs, self.params,
                self._simulation_status_bar, self._simulation_status_contents,
                self.connectivity_widgets, self.viz_manager)

        self.widget_backend_selection.observe(_handle_backend_change, 'value')
        self.add_drive_button.on_click(_add_drive_button_clicked)
        self.delete_drive_button.on_click(_delete_drives_clicked)
        self.load_connectivity_button.observe(_on_upload_connectivity,
                                              names='value')
        self.load_drives_button.observe(_on_upload_drives, names='value')
        self.run_button.on_click(_run_button_clicked)

    def compose(self, return_layout=True):
        """Compose widgets.

        Parameters
        ----------
        return_layout : bool
            If the method returns the layout object which can be rendered by
            IPython.display.display() method.
        """
        simulation_box = VBox([
            VBox([
                self.widget_simulation_name, self.widget_tstop, self.widget_dt,
                self.widget_ntrials, self.widget_backend_selection,
                self._backend_config_out
            ]),
        ], layout=self.layout['config_box'])

        connectivity_box = VBox([
            HBox([self.load_connectivity_button, ]),
            self._connectivity_out,
        ])

        # accordians to group local-connectivity by cell type
        connectivity_boxes = [
            VBox(slider) for slider in self.connectivity_widgets]
        connectivity_names = (
            'Layer 2/3 Pyramidal', 'Layer 5 Pyramidal', 'Layer 2 Basket',
            'Layer 5 Basket')
        cell_connectivity = Accordion(children=connectivity_boxes)
        for idx, connectivity_name in enumerate(connectivity_names):
            cell_connectivity.set_title(idx, connectivity_name)

        drive_selections = VBox([
            self.add_drive_button, self.widget_drive_type_selection,
            self.widget_location_selection],
            layout=Layout(flex="1"))

        drives_options = VBox([
            HBox([
                VBox([self.load_drives_button, self.delete_drive_button],
                     layout=Layout(flex="1")),
                drive_selections,
            ]), self._drives_out
        ])

        config_panel, figs_output = self.viz_manager.compose()

        # Tabs for left pane
        left_tab = Tab()
        left_tab.children = [
            simulation_box, connectivity_box, drives_options,
            config_panel,
        ]
        titles = ('Simulation', 'Network connectivity', 'External drives',
                  'Visualization')
        for idx, title in enumerate(titles):
            left_tab.set_title(idx, title)

        self.app_layout = AppLayout(
            header=self._header,
            left_sidebar=VBox([
                VBox([left_tab], layout=self.layout['left_tab']),
                self._operation_buttons,
                self._log_window,
            ], layout=self.layout['left_sidebar']),
            right_sidebar=figs_output,
            footer=self._simulation_status_bar,
            pane_widths=[
                self.layout['left_sidebar'].width, '0px',
                self.layout['visualization_window'].width
            ],
            pane_heights=[
                self.layout['header_height'],
                self.layout['visualization_window'].height,
                self.layout['simulation_status_height']
            ],
        )

        self._link_callbacks()

        # self.simulation_data[self.widget_simulation_name.value]

        # initialize drive and connectivity ipywidgets
        load_drive_and_connectivity(self.params, self._log_out,
                                    self._drives_out, self.drive_widgets,
                                    self.drive_boxes, self._connectivity_out,
                                    self.connectivity_widgets,
                                    self.widget_tstop, self.layout)

        if not return_layout:
            return
        else:
            return self.app_layout

    def show(self):
        display(self.app_layout)

    def capture(self, width=None, height=None, extra_margin=100, render=True):
        """Take a screenshot of the current GUI.

        Parameters
        ----------
        width : int | None
            The width of iframe window use to show the snapshot.
        height : int | None
            The height of iframe window use to show the snapshot.
        extra_margin: int
            Extra margin in pixel for the GUI.
        render : bool
            Will return an IFrame object if False

        Returns
        -------
        snapshot : An iframe snapshot object that can be rendered in notebooks.
        """
        file = io.StringIO()
        embed_minimal_html(file, views=[self.app_layout], title='')
        if not width:
            width = self.total_width + extra_margin
        if not height:
            height = self.total_height + extra_margin

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
    def _simulate_upload_connectivity(self, file_url):
        uploaded_value = _prepare_upload_file_from_url(file_url)
        self.load_connectivity_button.set_trait('value', uploaded_value)

    def _simulate_upload_drives(self, file_url):
        uploaded_value = _prepare_upload_file_from_url(file_url)
        self.load_drives_button.set_trait('value', uploaded_value)

    def _simulate_left_tab_click(self, tab_title):
        tab_index = None
        left_tab = self.app_layout.left_sidebar.children[0].children[0]
        for idx in left_tab._titles.keys():
            if tab_title == left_tab._titles[idx]:
                tab_index = int(idx)
                break
        if tab_index is None:
            raise ValueError("Incorrect tab title")
        left_tab.selected_index = tab_index

    def _simulate_make_figure(self,):
        self._simulate_left_tab_click("Visualization")
        self.viz_manager.make_fig_button.click()

    def _simulate_viz_action(self, action_name, *args, **kwargs):
        """A shortcut to call simulated actions in _VizManager.

        Parameters
        ----------
        action_name : str
            The action to take. For example, to call `_simulate_add_fig` in
            _VizManager, you can run `_simulate_viz_action("add_fig")`
        args : list
            Optional positional parameters passed to the called method.
        kwargs: dict
            Optional keyword parameters passed to the called method.
        """
        self._simulate_left_tab_click("Visualization")
        action = getattr(self.viz_manager, f"_simulate_{action_name}")
        action(*args, **kwargs)


def _prepare_upload_file_from_url(file_url):
    params_name = file_url.split("/")[-1]
    data = urllib.request.urlopen(file_url)
    content = b""
    for line in data:
        content += line

    return {
        params_name: {
            'metadata': {
                'name': params_name,
                'type': 'application/json',
                'size': len(content),
            },
            'content': content
        }
    }


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
                                 step=0.01, description="weight", style=style)

        conn_widget = VBox([
            HTML(value=f"""<p>
            Receptor: {conn_data[receptor_name]['receptor']}</p>"""),
            w_text_input, HTML(value="<hr style='margin-bottom:5px'/>")
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


def add_connectivity_tab(params, connectivity_out,
                         connectivity_textfields):
    """Add all possible connectivity boxes to connectivity tab."""
    net = jones_2009_model(params)
    cell_types = [ct for ct in net.cell_types.keys()]
    receptors = ('ampa', 'nmda', 'gabaa', 'gabab')
    locations = ('proximal', 'distal', 'soma')

    # clear existing connectivity
    connectivity_out.clear_output()
    while len(connectivity_textfields) > 0:
        connectivity_textfields.pop()

    connectivity_names = list()
    for src_gids in cell_types:
        for target_gids in cell_types:
            for location in locations:
                # the connectivity list should be built on this level
                receptor_related_conn = {}
                for receptor in receptors:
                    conn_indices = pick_connection(net=net,
                                                   src_gids=src_gids,
                                                   target_gids=target_gids,
                                                   loc=location,
                                                   receptor=receptor)
                    if len(conn_indices) > 0:
                        assert len(conn_indices) == 1
                        conn_idx = conn_indices[0]
                        current_w = net.connectivity[
                            conn_idx]['nc_dict']['A_weight']
                        current_p = net.connectivity[
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
                    connectivity_textfields.append(
                        _get_connectivity_widgets(receptor_related_conn))

    connectivity_boxes = [VBox(slider) for slider in connectivity_textfields]
    cell_connectivity = Accordion(children=connectivity_boxes)
    for idx, connectivity_name in enumerate(connectivity_names):
        cell_connectivity.set_title(idx, connectivity_name)

    with connectivity_out:
        display(cell_connectivity)

    return net


def add_drive_tab(params, drives_out, drive_widgets, drive_boxes, tstop,
                  layout):
    net = jones_2009_model(params)
    drive_specs = _extract_drive_specs_from_hnn_params(
        params,
        list(net.cell_types.keys()))

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


def load_drive_and_connectivity(params, log_out, drives_out,
                                drive_widgets, drive_boxes, connectivity_out,
                                connectivity_textfields, tstop, layout):
    """Add drive and connectivity ipywidgets from params."""
    log_out.clear_output()
    with log_out:
        # Add connectivity
        add_connectivity_tab(params, connectivity_out, connectivity_textfields)
        # Add drives
        add_drive_tab(params, drives_out, drive_widgets, drive_boxes, tstop,
                      layout)


def on_upload_change(change, params, tstop, dt, log_out, drive_boxes,
                     drive_widgets, drives_out, connectivity_out,
                     connectivity_textfields, layout, load_type):
    if len(change['owner'].value) == 0:
        logger.info("Empty change")
        return
    logger.info("Loading connectivity...")
    key = list(change['new'].keys())[0]

    params_fname = change['new'][key]['metadata']['name']
    param_data = change['new'][key]['content']

    param_data = codecs.decode(param_data, encoding="utf-8")

    ext = Path(params_fname).suffix
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
    if load_type == 'connectivity':
        add_connectivity_tab(params, connectivity_out, connectivity_textfields)
    elif load_type == 'drives':
        add_drive_tab(params, drives_out, drive_widgets, drive_boxes, tstop,
                      layout)
    else:
        raise ValueError

    change['owner'].set_trait('_counter', 0)
    change['owner'].set_trait('value', {})


def _init_network_from_widgets(params, dt, tstop, single_simulation_data,
                               drive_widgets, connectivity_textfields,
                               add_drive=True):
    """Construct network and add drives."""
    print("init network")
    params['dt'] = dt.value
    params['tstop'] = tstop.value
    single_simulation_data['net'] = jones_2009_model(
        params,
        add_drives_from_params=False,
    )
    # adjust connectivity according to the connectivity_tab
    for connectivity_slider in connectivity_textfields:
        for vbox in connectivity_slider:
            conn_indices = pick_connection(
                net=single_simulation_data['net'],
                src_gids=vbox._belongsto['src_gids'],
                target_gids=vbox._belongsto['target_gids'],
                loc=vbox._belongsto['location'],
                receptor=vbox._belongsto['receptor'])

            if len(conn_indices) > 0:
                assert len(conn_indices) == 1
                conn_idx = conn_indices[0]
                single_simulation_data['net'].connectivity[conn_idx][
                    'nc_dict']['A_weight'] = vbox.children[1].value
                single_simulation_data['net'].connectivity[conn_idx][
                    'probability'] = vbox.children[2].value

    if add_drive is False:
        return
    # add drives to network
    for drive in drive_widgets:
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
            single_simulation_data['net'].add_poisson_drive(
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
            single_simulation_data['net'].add_evoked_drive(
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
            single_simulation_data['net'].add_bursty_drive(
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


def run_button_clicked(widget_simulation_name, log_out, drive_widgets,
                       all_data, dt, tstop, ntrials, backend_selection,
                       mpi_cmd, n_jobs, params, simulation_status_bar,
                       simulation_status_contents, connectivity_textfields,
                       viz_manager):
    """Run the simulation and plot outputs."""
    log_out.clear_output()
    simulation_data = all_data["simulation_data"]
    with log_out:
        # clear empty trash simulations
        for _name in tuple(simulation_data.keys()):
            if simulation_data[_name]['net'] is None or len(
                    simulation_data[_name]['dpls']) == 0:
                del simulation_data[_name]

        _sim_name = widget_simulation_name.value
        if simulation_data[_sim_name]['net'] is not None:
            print("Simulation with the same name exists!")
            simulation_status_bar.value = simulation_status_contents[
                'failed']
            return

        _init_network_from_widgets(params, dt, tstop,
                                   simulation_data[_sim_name], drive_widgets,
                                   connectivity_textfields)

        print("start simulation")
        if backend_selection.value == "MPI":
            backend = MPIBackend(
                n_procs=multiprocessing.cpu_count() - 1, mpi_cmd=mpi_cmd.value)
        else:
            backend = JoblibBackend(n_jobs=n_jobs.value)
            print(f"Using Joblib with {n_jobs.value} core(s).")
        with backend:
            simulation_status_bar.value = simulation_status_contents['running']
            simulation_data[_sim_name]['dpls'] = simulate_dipole(
                simulation_data[_sim_name]['net'],
                tstop=tstop.value,
                dt=dt.value,
                n_trials=ntrials.value)

            simulation_status_bar.value = simulation_status_contents[
                'finished']

    viz_manager.reset_fig_config_tabs()
    viz_manager.add_figure()
    fig_name = _idx2figname(viz_manager.data['fig_idx']['idx'] - 1)
    ax_plots = [("ax0", "input histogram"), ("ax1", "current dipole")]
    for ax_name, plot_type in ax_plots:
        viz_manager._simulate_edit_figure(fig_name, ax_name, _sim_name,
                                          plot_type, {}, "plot")


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
    notebook_path = Path(__file__).parent / 'hnn_widget.ipynb'
    main([str(notebook_path.resolve()), *sys.argv[1:]])
