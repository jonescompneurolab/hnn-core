"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>
import base64
import codecs
import io
import logging
import mimetypes
import numpy as np
import sys
import json
import re
import textwrap
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from functools import partial
from IPython.display import IFrame, display
from ipywidgets import (
    HTML,
    Accordion,
    AppLayout,
    FloatText,
    BoundedFloatText,
    BoundedIntText,
    Button,
    Dropdown,
    FileUpload,
    VBox,
    HBox,
    IntText,
    Layout,
    Output,
    RadioButtons,
    Tab,
    Text,
    Checkbox,
    Box,
)
from ipywidgets.embed import embed_minimal_html
import hnn_core
from hnn_core import JoblibBackend, MPIBackend, simulate_dipole
from hnn_core.optimization import Optimizer
from hnn_core.gui._logging import logger
from hnn_core.gui._viz_manager import _VizManager, _idx2figname
from hnn_core.network import pick_connection
from hnn_core.dipole import _read_dipole_txt
from hnn_core.params_default import get_L2Pyr_params_default, get_L5Pyr_params_default
from hnn_core.hnn_io import dict_to_network, write_network_configuration
from hnn_core.cells_default import _exp_g_at_dist
from hnn_core.parallel_backends import (
    _determine_cores_hwthreading,
    _has_mpi4py,
    _has_psutil,
)

hnn_core_root = Path(hnn_core.__file__).parent
default_network_configuration = hnn_core_root / "param" / "jones2009_base.json"

cell_parameters_dict = {
    "Geometry L2": [
        ("Soma length", "micron", "soma_L"),
        ("Soma diameter", "micron", "soma_diam"),
        ("Soma capacitive density", "F/cm2", "soma_cm"),
        ("Soma resistivity", "ohm-cm", "soma_Ra"),
        ("Dendrite capacitive density", "F/cm2", "dend_cm"),
        ("Dendrite resistivity", "ohm-cm", "dend_Ra"),
        ("Apical Dendrite Trunk length", "micron", "apicaltrunk_L"),
        ("Apical Dendrite Trunk diameter", "micron", "apicaltrunk_diam"),
        ("Apical Dendrite 1 length", "micron", "apical1_L"),
        ("Apical Dendrite 1 diameter", "micron", "apical1_diam"),
        ("Apical Dendrite Tuft length", "micron", "apicaltuft_L"),
        ("Apical Dendrite Tuft diameter", "micron", "apicaltuft_diam"),
        ("Oblique Apical Dendrite length", "micron", "apicaloblique_L"),
        ("Oblique Apical Dendrite diameter", "micron", "apicaloblique_diam"),
        ("Basal Dendrite 1 length", "micron", "basal1_L"),
        ("Basal Dendrite 1 diameter", "micron", "basal1_diam"),
        ("Basal Dendrite 2 length", "micron", "basal2_L"),
        ("Basal Dendrite 2 diameter", "micron", "basal2_diam"),
        ("Basal Dendrite 3 length", "micron", "basal3_L"),
        ("Basal Dendrite 3 diameter", "micron", "basal3_diam"),
    ],
    "Geometry L5": [
        ("Soma length", "micron", "soma_L"),
        ("Soma diameter", "micron", "soma_diam"),
        ("Soma capacitive density", "F/cm2", "soma_cm"),
        ("Soma resistivity", "ohm-cm", "soma_Ra"),
        ("Dendrite capacitive density", "F/cm2", "dend_cm"),
        ("Dendrite resistivity", "ohm-cm", "dend_Ra"),
        ("Apical Dendrite Trunk length", "micron", "apicaltrunk_L"),
        ("Apical Dendrite Trunk diameter", "micron", "apicaltrunk_diam"),
        ("Apical Dendrite 1 length", "micron", "apical1_L"),
        ("Apical Dendrite 1 diameter", "micron", "apical1_diam"),
        ("Apical Dendrite 2 length", "micron", "apical2_L"),
        ("Apical Dendrite 2 diameter", "micron", "apical2_diam"),
        ("Apical Dendrite Tuft length", "micron", "apicaltuft_L"),
        ("Apical Dendrite Tuft diameter", "micron", "apicaltuft_diam"),
        ("Oblique Apical Dendrite length", "micron", "apicaloblique_L"),
        ("Oblique Apical Dendrite diameter", "micron", "apicaloblique_diam"),
        ("Basal Dendrite 1 length", "micron", "basal1_L"),
        ("Basal Dendrite 1 diameter", "micron", "basal1_diam"),
        ("Basal Dendrite 2 length", "micron", "basal2_L"),
        ("Basal Dendrite 2 diameter", "micron", "basal2_diam"),
        ("Basal Dendrite 3 length", "micron", "basal3_L"),
        ("Basal Dendrite 3 diameter", "micron", "basal3_diam"),
    ],
    "Synapses": [
        ("AMPA reversal", "mV", "ampa_e"),
        ("AMPA rise time", "ms", "ampa_tau1"),
        ("AMPA decay time", "ms", "ampa_tau2"),
        ("NMDA reversal", "mV", "nmda_e"),
        ("NMDA rise time", "ms", "nmda_tau1"),
        ("NMDA decay time", "ms", "nmda_tau2"),
        ("GABAA reversal", "mV", "gabaa_e"),
        ("GABAA rise time", "ms", "gabaa_tau1"),
        ("GABAA decay time", "ms", "gabaa_tau2"),
        ("GABAB reversal", "mV", "gabab_e"),
        ("GABAB rise time", "ms", "gabab_tau1"),
        ("GABAB decay time", "ms", "gabab_tau2"),
    ],
    "Biophysics L2": [
        ("Soma Kv channel density", "S/cm2", "soma_gkbar_hh2"),
        ("Soma Na channel density", "S/cm2", "soma_gnabar_hh2"),
        ("Soma leak reversal", "mV", "soma_el_hh2"),
        ("Soma leak channel density", "S/cm2", "soma_gl_hh2"),
        ("Soma Km channel density", "pS/micron2", "soma_gbar_km"),
        ("Dendrite Kv channel density", "S/cm2", "dend_gkbar_hh2"),
        ("Dendrite Na channel density", "S/cm2", "dend_gnabar_hh2"),
        ("Dendrite leak reversal", "mV", "dend_el_hh2"),
        ("Dendrite leak channel density", "S/cm2", "dend_gl_hh2"),
        ("Dendrite Km channel density", "pS/micron2", "dend_gbar_km"),
    ],
    "Biophysics L5": [
        ("Soma Kv channel density", "S/cm2", "soma_gkbar_hh2"),
        ("Soma Na channel density", "S/cm2", "soma_gnabar_hh2"),
        ("Soma leak reversal", "mV", "soma_el_hh2"),
        ("Soma leak channel density", "S/cm2", "soma_gl_hh2"),
        ("Soma Ca channel density", "pS/micron2", "soma_gbar_ca"),
        ("Soma Ca decay time", "ms", "soma_taur_cad"),
        ("Soma Kca channel density", "pS/micron2", "soma_gbar_kca"),
        ("Soma Km channel density", "pS/micron2", "soma_gbar_km"),
        ("Soma CaT channel density", "S/cm2", "soma_gbar_cat"),
        ("Soma HCN channel density", "S/cm2", "soma_gbar_ar"),
        ("Dendrite Kv channel density", "S/cm2", "dend_gkbar_hh2"),
        ("Dendrite Na channel density", "S/cm2", "dend_gnabar_hh2"),
        ("Dendrite leak reversal", "mV", "dend_el_hh2"),
        ("Dendrite leak channel density", "S/cm2", "dend_gl_hh2"),
        ("Dendrite Ca channel density", "pS/micron2", "dend_gbar_ca"),
        ("Dendrite Ca decay time", "ms", "dend_taur_cad"),
        ("Dendrite KCa channel density", "pS/micron2", "dend_gbar_kca"),
        ("Dendrite Km channel density", "pS/micron2", "dend_gbar_km"),
        ("Dendrite CaT channel density", "S/cm2", "dend_gbar_cat"),
        ("Dendrite HCN channel density", "S/cm2", "dend_gbar_ar"),
    ],
}

global_gain_type_display_dict = {
    "e_e": "Exc-to-Exc",
    "e_i": "Exc-to-Inh",
    "i_e": "Inh-to-Exc",
    "i_i": "Inh-to-Inh",
}

global_gain_type_lookup_dict = {
    ("L2_pyramidal", "L2_pyramidal"): "e_e",
    ("L2_pyramidal", "L5_pyramidal"): "e_e",
    ("L5_pyramidal", "L5_pyramidal"): "e_e",
    ("L2_pyramidal", "L2_basket"): "e_i",
    ("L2_pyramidal", "L5_basket"): "e_i",
    ("L5_pyramidal", "L5_basket"): "e_i",
    ("L2_basket", "L2_pyramidal"): "i_e",
    ("L2_basket", "L5_pyramidal"): "i_e",
    ("L5_basket", "L5_pyramidal"): "i_e",
    ("L2_basket", "L2_basket"): "i_i",
    ("L5_basket", "L5_basket"): "i_i",
}


class _OutputWidgetHandler(logging.Handler):
    def __init__(self, output_widget, *args, **kwargs):
        super(_OutputWidgetHandler, self).__init__(*args, **kwargs)
        self.out = output_widget

    def emit(self, record):
        formatted_record = self.format(record)
        # Further format the message for GUI presentation
        try:
            formatted_record = formatted_record.replace("  - ", "\n")
            formatted_record = "[TIME] " + formatted_record + "\n"
        except:
            pass
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output,) + self.out.outputs


class _GUI_PrintToLogger:
    """Class to redirect print messages to the logger in the GUI"""

    # when print is used, call the write method instead
    def write(self, message):
        # avoid logging empty/new lines
        if message.strip():
            # send the message to the logger
            logger.info(message.strip())

    # The flush method is required for compatibility with print
    def flush(self):
        pass


# assign class to stdout to redirect print statements to the logger
sys.stdout = _GUI_PrintToLogger()


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

    def __init__(
        self,
        theme_color="#802989",
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
        network_configuration=default_network_configuration,
    ):
        # set up styling.
        self.total_height = total_height
        self.total_width = total_width

        viz_win_width = self.total_width - left_sidebar_width
        main_content_height = self.total_height - status_height

        config_box_height = main_content_height - (
            log_window_height + operation_box_height
        )
        self.layout = {
            "dpi": dpi,
            "header_height": f"{header_height}px",
            "theme_color": theme_color,
            "btn": Layout(height=f"{button_height}px", width="auto"),
            "run_btn": Layout(height=f"{button_height}px", width="10%"),
            "btn_full_w": Layout(height=f"{button_height}px", width="100%"),
            "del_fig_btn": Layout(height=f"{button_height}px", width="auto"),
            "log_out": Layout(
                border="1px solid gray",
                height=f"{log_window_height - 10}px",
                overflow="auto",
            ),
            "viz_config": Layout(width="99%"),
            "simulations_list": Layout(width=f"{left_sidebar_width - 50}px"),
            "visualization_window": Layout(
                width=f"{viz_win_width - 10}px",
                height=f"{main_content_height - 10}px",
                border="1px solid gray",
                overflow="scroll",
            ),
            "visualization_output": Layout(
                width=f"{viz_win_width - 50}px",
                height=f"{main_content_height - 100}px",
                border="1px solid gray",
                overflow="scroll",
            ),
            "left_sidebar": Layout(
                width=f"{left_sidebar_width}px", height=f"{main_content_height}px"
            ),
            "left_tab": Layout(
                width=f"{left_sidebar_width}px", height=f"{config_box_height}px"
            ),
            "operation_box": Layout(
                width=f"{left_sidebar_width}px",
                height=f"{operation_box_height}px",
                flex_wrap="wrap",
            ),
            "config_box": Layout(
                width=f"{left_sidebar_width - 40}px",
                height=f"{config_box_height - 100}px",
            ),
            "drive_widget": Layout(width="auto"),
            "drive_textbox": Layout(width="270px", height="auto"),
            # optimization related
            "opt_textbox": Layout(width="250px"),
            # simulation status related
            "simulation_status_height": f"{status_height}px",
            "simulation_status_common": "background:gray;padding-left:10px",
            "simulation_status_running": "background:orange;padding-left:10px",
            "simulation_status_failed": "background:red;padding-left:10px",
            "simulation_status_finished": "background:green;padding-left:10px",
        }

        self._simulation_status_contents = {
            "not_running": f"""<div style='{self.layout["simulation_status_common"]};
            color:white;'>Not running</div>""",
            "running": f"""<div style='{self.layout["simulation_status_running"]};
            color:white;'>Running...</div>""",
            "opt_running": f"""<div style='{self.layout["simulation_status_running"]};
            color:white;'>Optimization Running, please be patient...</div>""",
            "finished": f"""<div style='{self.layout["simulation_status_finished"]};
            color:white;'>Simulation finished</div>""",
            "failed": f"""<div style='{self.layout["simulation_status_failed"]};
            color:white;'>Simulation failed</div>""",
        }

        # load default parameters
        self.params = self.load_parameters(network_configuration)

        # Number of available cores
        [self.n_cores, _] = _determine_cores_hwthreading(
            use_hwthreading_if_found=False,
            sensible_default_cores=True,
        )

        # In-memory storage of all simulation and visualization related data
        self.simulation_data = defaultdict(lambda: dict(net=None, dpls=list()))

        # Default visualization params for figures
        analysis_style = {"description_width": "200px"}
        layout = Layout(width="300px")

        self.widget_default_smoothing = BoundedFloatText(
            value=30.0,
            description="Dipole Smoothing:",
            min=0.0,
            max=100.0,
            step=1.0,
            disabled=False,
            layout=layout,
            style=analysis_style,
        )

        self.widget_default_scaling = FloatText(
            value=3000.0,
            description="Dipole Scaling:",
            step=100.0,
            disabled=False,
            layout=layout,
            style=analysis_style,
        )

        self.widget_min_frequency = BoundedFloatText(
            value=10,
            min=0.1,
            max=1000,
            description="Min Spectral Frequency (Hz):",
            disabled=False,
            layout=layout,
            style=analysis_style,
        )

        self.widget_max_frequency = BoundedFloatText(
            value=100,
            min=0.1,
            max=1000,
            description="Max Spectral Frequency (Hz):",
            disabled=False,
            layout=layout,
            style=analysis_style,
        )

        self.fig_default_params = {
            "default_smoothing": self.widget_default_smoothing.value,
            "default_scaling": self.widget_default_scaling.value,
            "default_min_frequency": self.widget_min_frequency.value,
            "default_max_frequency": self.widget_max_frequency.value,
        }

        # Simulation parameters
        self.widget_tstop = BoundedFloatText(
            value=170, description="tstop (ms):", min=0, max=1e6, step=1, disabled=False
        )
        self.widget_dt = BoundedFloatText(
            value=0.025,
            description="dt (ms):",
            min=0,
            max=10,
            step=0.01,
            disabled=False,
        )
        self.widget_ntrials = IntText(value=1, description="Trials:", disabled=False)
        self.widget_simulation_name = Text(
            value="default",
            placeholder="ID of your simulation",
            description="Name:",
            disabled=False,
        )
        self.widget_backend_selection = Dropdown(
            options=[("Joblib", "Joblib"), ("MPI", "MPI")],
            value=self._check_backend(),
            description="Backend:",
        )
        self.widget_mpi_cmd = Text(
            value="mpiexec",
            placeholder="Fill if applies",
            description="MPI cmd:",
            disabled=False,
        )
        self.widget_n_jobs = BoundedIntText(
            value=1, min=1, max=self.n_cores, description="Cores:", disabled=False
        )
        self.load_data_button = FileUpload(
            accept=".txt,.csv",
            multiple=False,
            style={"button_color": self.layout["theme_color"]},
            layout=self.layout["btn"],
            description="Load data",
            button_style="success",
        )

        # Create save simulation widget wrapper
        self.save_simuation_button = self._init_html_download_button(
            title="Save Simulation", mimetype="text/csv"
        )
        self.save_config_button = self._init_html_download_button(
            title="Save Network", mimetype="application/json"
        )

        self.simulation_list_widget = Dropdown(
            options=[], value=None, description="", layout={"width": "15%"}
        )

        # Drive selection
        drive_dropdown_style = {"description_width": "100px"}

        self.widget_drive_type_selection = Dropdown(
            options=["Evoked", "Poisson", "Rhythmic", "Tonic"],
            value="Evoked",
            description="Drive type:",
            disabled=False,
            layout=self.layout["drive_widget"],
            style=drive_dropdown_style,
        )
        self.widget_location_selection = Dropdown(
            options=["Proximal", "Distal"],
            value="Proximal",
            description="Drive location:",
            disabled=False,
            layout=self.layout["drive_widget"],
            style=drive_dropdown_style,
        )
        self.add_drive_button = create_expanded_button(
            "Add drive",
            "primary",
            layout=self.layout["btn"],
            button_color=self.layout["theme_color"],
        )

        # Optimizer widgets
        # Just use same styling as top-level drive widgets (not accordion)
        opt_dropdown_style = {"description_width": "120px"}

        self.widget_opt_solver = Dropdown(
            options=["bayesian", "cobyla"],
            value="bayesian",
            description="Solver:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_obj_fun = Dropdown(
            options=["dipole_rmse", "maximize_psd"],
            value="dipole_rmse",
            description="Objective Function:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_max_iter = BoundedIntText(
            # value=200,  # AES debug
            value=3,
            # value=15,
            min=1,
            max=10000,
            description="Max Iterations:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_tstop = BoundedFloatText(
            value=170,
            min=0.1,
            max=1000.0,
            description="tstop (ms):",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )

        # # DATA to optimize towards aka target data
        # # Note: all data, included loaded/experimental data, seems to be governed under
        # # "self.simulation_data".
        # #
        # # AES UGH need to dive in and debug with pytest to figure out HOW DO I ACCESS LOADED DATA
        # sim_names = [
        #     simulations
        #     for simulations, sim_name in self.data["simulation_data"].items()
        #     if sim_name["net"] is not None
        # ]
        # if len(sim_names) == 0:
        #     sim_names = [" "]

        self.widget_opt_target_data = Dropdown(
            # options=self.simulation_data,
            # options=self.data,  # this at least prints "simulation_data"
            # options=self.data["simulation_data"],  # nope
            # options=self.data["simulation_data"],  # nope
            options=self.data["simulation_data"].keys(),  # nope
            # options=self.viz_manager.datasets_dropdown.options,  # nope
            # options=sim_names,
            # options=None,
            # value=sim_names[0],
            value=None,
            description="Target Data:",
            disabled=False,
            layout=Layout(width="98%"),
        )

        self.run_opt_button = create_expanded_button(
            "Run Optimization",
            "success",
            layout=Layout(width="auto"),
            button_color=self.layout["theme_color"],
        )

        # Dashboard level buttons
        self.run_button = create_expanded_button(
            "Run",
            "success",
            layout=self.layout["run_btn"],
            button_color=self.layout["theme_color"],
        )

        self.load_connectivity_button = FileUpload(
            accept=".json",
            multiple=False,
            style={"button_color": self.layout["theme_color"]},
            description="Load local network connectivity",
            layout=self.layout["btn_full_w"],
            button_style="success",
        )
        self.load_drives_button = FileUpload(
            accept=".json",
            multiple=False,
            style={"button_color": self.layout["theme_color"]},
            description="Load external drives",
            layout=self.layout["btn"],
            button_style="success",
        )

        self.delete_drive_button = create_expanded_button(
            "Delete all drives",
            "success",
            layout=self.layout["btn"],
            button_color=self.layout["theme_color"],
        )

        self.cell_type_radio_buttons = RadioButtons(
            options=["L2/3 Pyramidal", "L5 Pyramidal"], description="Cell type:"
        )

        self.cell_layer_radio_buttons = RadioButtons(
            options=["Geometry", "Synapses", "Biophysics"],
            description="Cell Properties:",
        )

        # Plotting window

        # Visualization figure related dicts
        self.plot_outputs_dict = dict()
        self.plot_dropdown_types_dict = dict()
        self.plot_sim_selections_dict = dict()

        # Add drive section
        self.drive_widgets = list()
        self.drive_boxes = list()
        self.drive_accordion = Accordion()

        # Connectivity list
        self.connectivity_widgets = list()

        # Cell parameter dict
        self.cell_parameters_widgets = dict()

        # Synaptic Gains dict
        self.global_gain_widgets = dict()

        # Add optimzation section
        self.opt_drive_widgets = list()
        self.opt_drive_boxes = list()
        self.opt_accordion = Accordion()

        self._init_ui_components()
        self.add_logging_window_logger()

    @staticmethod
    def _check_backend():
        """Checks for MPI and returns the default backend name"""
        default_backend = "Joblib"
        if _has_mpi4py() and _has_psutil():
            default_backend = "MPI"
        return default_backend

    def get_cell_parameters_dict(self):
        """Returns the number of elements in the
        cell_parameters_dict dictionary.
        This is for testing purposes"""
        return cell_parameters_dict

    def _init_html_download_button(self, title, mimetype):
        b64 = base64.b64encode("".encode())
        payload = b64.decode()
        # Initialliting HTML code for download button
        self.html_download_button = """
        <a download="{filename}" href="data:{mimetype};base64,{payload}"
          download>
        <button style="background:{color_theme}; height:{btn_height}"
        class=" jupyter-button
           mod-warning" {is_disabled} >{title}</button>
        </a>
        """
        # Create widget wrapper
        return HTML(
            self.html_download_button.format(
                payload=payload,
                filename={""},
                is_disabled="disabled",
                btn_height=self.layout["run_btn"].height,
                color_theme=self.layout["theme_color"],
                title=title,
                mimetype=mimetype,
            )
        )

    def add_logging_window_logger(self):
        handler = _OutputWidgetHandler(self._log_out)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  - [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)

    def _init_ui_components(self):
        """Initialize larger UI components and dynamical output windows.

        It's not encouraged for users to modify or access attributes in this
        part.
        """
        # dynamic larger components
        self._drives_out = Output()  # tab to add new drives
        self._connectivity_out = Output()  # tab to tune connectivity.
        self._cell_params_out = Output()
        self._global_gain_out = Output()
        self._opt_out = Output()  # dynamic part of optimization tab

        self._log_out = Output()

        self.viz_manager = _VizManager(self.data, self.layout, self.fig_default_params)

        # Register widget_opt_target_data to be updated when simulation data changes
        self.viz_manager._external_data_widget = self.widget_opt_target_data

        # detailed configuration of backends
        self._backend_config_out = Output()

        # static parts
        # Running status
        self._simulation_status_bar = HTML(
            value=self._simulation_status_contents["not_running"]
        )

        self._log_window = HBox([self._log_out], layout=self.layout["log_out"])
        self._operation_buttons = HBox(
            [
                self.run_button,
                self.load_data_button,
                self.save_config_button,
                self.save_simuation_button,
                self.simulation_list_widget,
            ],
            layout=self.layout["operation_box"],
        )
        # title
        self._header = HTML(
            value=f"""<div
            style='background:{self.layout["theme_color"]};
            text-align:center;color:white;'>
            HUMAN NEOCORTICAL NEUROSOLVER</div>"""
        )

    @property
    def analysis_config(self):
        """Provides everything viz window needs except for the data."""
        return {
            "viz_style": self.layout["visualization_output"],
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
    def load_parameters(params_fname):
        """Read parameters from file."""
        with open(params_fname, "r") as file:
            parameters = json.load(file)

        return parameters

    def _link_callbacks(self):
        """Link callbacks to UI components."""

        def _handle_backend_change(backend_type):
            return handle_backend_change(
                backend_type.new,
                self._backend_config_out,
                self.widget_mpi_cmd,
                self.widget_n_jobs,
            )

        def _add_drive_button_clicked(b):
            location = self.widget_location_selection.value.lower()
            return self.add_drive_widget(
                self.widget_drive_type_selection.value,
                location,
            )

        def _delete_drives_clicked(b):
            self._drives_out.clear_output()
            # black magic: the following does not work
            # global drive_widgets; drive_widgets = list()
            while len(self.drive_widgets) > 0:
                self.drive_widgets.pop()
                self.drive_boxes.pop()

        def _on_upload_connectivity(change):
            new_params = self.on_upload_params_change(
                change, self.layout["drive_textbox"], load_type="connectivity"
            )
            self.params = new_params

        def _on_upload_drives(change):
            _ = self.on_upload_params_change(
                change, self.layout["drive_textbox"], load_type="drives"
            )

        def _on_upload_data(change):
            return on_upload_data_change(
                change, self.data, self.viz_manager, self._log_out
            )

        def _run_button_clicked(b):
            return run_button_clicked(
                self.widget_simulation_name,
                self._log_out,
                self.drive_widgets,
                self.data,
                self.widget_dt,
                self.widget_tstop,
                self.fig_default_params,
                self.widget_default_smoothing,
                self.widget_default_scaling,
                self.widget_min_frequency,
                self.widget_max_frequency,
                self.widget_ntrials,
                self.widget_backend_selection,
                self.widget_mpi_cmd,
                self.widget_n_jobs,
                self.params,
                self._simulation_status_bar,
                self._simulation_status_contents,
                self.connectivity_widgets,
                self.viz_manager,
                self.simulation_list_widget,
                self.cell_parameters_widgets,
                self.global_gain_widgets,
            )

        def _run_opt_button_clicked(b):
            return run_opt_button_clicked(
                self.widget_simulation_name,
                self._log_out,
                self.opt_drive_widgets,
                self.data,
                self.widget_dt,
                self.widget_tstop,
                self.fig_default_params,
                self.widget_default_smoothing,
                self.widget_default_scaling,
                self.widget_min_frequency,
                self.widget_max_frequency,
                self.widget_ntrials,
                self.widget_backend_selection,
                self.widget_mpi_cmd,
                self.widget_n_jobs,
                self.params,
                self._simulation_status_bar,
                self._simulation_status_contents,
                self.connectivity_widgets,
                self.viz_manager,
                self.simulation_list_widget,
                self.cell_parameters_widgets,
                self.global_gain_widgets,
                self.widget_opt_solver.value,
                self.widget_opt_obj_fun.value,
                self.widget_opt_max_iter.value,
                self.widget_opt_tstop.value,
                self.widget_opt_target_data.value,
            )

        def _simulation_list_change(value):
            # Simulation Data
            _simulation_data, file_extension = _serialize_simulation(
                self._log_out, self.data, self.simulation_list_widget
            )

            result_file = f"{value.new}{file_extension}"
            if file_extension == ".csv":
                b64 = base64.b64encode(_simulation_data.encode())
            else:
                b64 = base64.b64encode(_simulation_data)

            payload = b64.decode()
            self.save_simuation_button.value = self.html_download_button.format(
                payload=payload,
                filename=result_file,
                is_disabled="",
                btn_height=self.layout["run_btn"].height,
                color_theme=self.layout["theme_color"],
                title="Save Simulation",
                mimetype="text/csv",
            )

            # Network Configuration
            network_config = _serialize_config(
                self._log_out, self.data, self.simulation_list_widget
            )
            b64_net = base64.b64encode(network_config.encode())
            self.save_config_button.value = self.html_download_button.format(
                payload=b64_net.decode(),
                filename=f"{value.new}.json",
                is_disabled="",
                btn_height=self.layout["run_btn"].height,
                color_theme=self.layout["theme_color"],
                title="Save Network",
                mimetype="application/json",
            )

        def _driver_type_change(value):
            self.widget_location_selection.disabled = (
                True if value.new == "Tonic" else False
            )

        def _cell_type_radio_change(value):
            _update_cell_params_vbox(
                self._cell_params_out,
                self.cell_parameters_widgets,
                value.new,
                self.cell_layer_radio_buttons.value,
            )

        def _cell_layer_radio_change(value):
            _update_cell_params_vbox(
                self._cell_params_out,
                self.cell_parameters_widgets,
                self.cell_type_radio_buttons.value,
                value.new,
            )

        self.widget_backend_selection.observe(_handle_backend_change, "value")
        self.add_drive_button.on_click(_add_drive_button_clicked)
        self.delete_drive_button.on_click(_delete_drives_clicked)
        self.load_connectivity_button.observe(_on_upload_connectivity, names="value")
        self.load_drives_button.observe(_on_upload_drives, names="value")
        self.run_button.on_click(_run_button_clicked)
        self.run_opt_button.on_click(_run_opt_button_clicked)

        self.load_data_button.observe(_on_upload_data, names="value")
        self.simulation_list_widget.observe(_simulation_list_change, "value")
        self.widget_drive_type_selection.observe(_driver_type_change, "value")

        self.cell_type_radio_buttons.observe(_cell_type_radio_change, "value")
        self.cell_layer_radio_buttons.observe(_cell_layer_radio_change, "value")

        # AES why isn't this working
        # self.widget_opt_target_data.observe(_on_upload_data, names="value")
        # self.widget_opt_target_data.observe(self.viz_manager._layout_template_change, names="value")

    def _delete_single_drive(self, b):
        index = self.drive_accordion.selected_index

        # Remove selected drive from drive lists
        self.drive_boxes.pop(index)
        self.drive_widgets.pop(index)

        # Rebuild the accordion collection
        self.drive_accordion.titles = tuple(
            t for i, t in enumerate(self.drive_accordion.titles) if i != index
        )
        self.drive_accordion.selected_index = None
        self.drive_accordion.children = self.drive_boxes

        # Render
        self._drives_out.clear_output()
        with self._drives_out:
            display(self.drive_accordion)

    def compose(self, return_layout=True):
        """Compose widgets.

        Parameters
        ----------
        return_layout : bool
            If the method returns the layout object which can be rendered by
            IPython.display.display() method.
        """
        box_style = """
            style="
                background: gray;
                color: white;
                # font-weight: bold;
                width: 290px;
                padding: 0px 5px;
                margin-bottom: 2px;
            "
        """
        simulation_box = VBox(
            [
                HTML(f"<div {box_style}>Simulation Parameters</div>"),
                VBox(
                    [
                        self.widget_simulation_name,
                        self.widget_tstop,
                        self.widget_dt,
                        self.widget_ntrials,
                        self.widget_backend_selection,
                        self._backend_config_out,
                    ]
                ),
                Box(layout=Layout(height="20px")),
                HTML(
                    f"<div {box_style}'>Default Visualization Parameters</div>",
                ),
                VBox(
                    [
                        self.widget_default_smoothing,
                        self.widget_default_scaling,
                        self.widget_min_frequency,
                        self.widget_max_frequency,
                    ]
                ),
            ],
            layout=self.layout["config_box"],
        )
        # Displays the default backend options
        handle_backend_change(
            self.widget_backend_selection.value,
            self._backend_config_out,
            self.widget_mpi_cmd,
            self.widget_n_jobs,
        )

        connectivity_configuration = Tab()

        connectivity_box = VBox(
            [
                HBox(
                    [
                        self.load_connectivity_button,
                    ]
                ),
                VBox(
                    [
                        self._global_gain_out,
                    ]
                ),
                self._connectivity_out,
            ]
        )

        cell_parameters = VBox(
            [
                HBox([self.cell_type_radio_buttons, self.cell_layer_radio_buttons]),
                self._cell_params_out,
            ]
        )

        connectivity_configuration.children = [
            connectivity_box,
            cell_parameters,
        ]
        connectivity_configuration.titles = [
            "Connectivity",
            "Cell parameters",
        ]

        drive_selections = VBox(
            [
                self.add_drive_button,
                self.widget_drive_type_selection,
                self.widget_location_selection,
            ],
            layout=Layout(flex="1"),
        )

        drives_options = VBox(
            [
                HBox(
                    [
                        VBox(
                            [self.load_drives_button, self.delete_drive_button],
                            layout=Layout(flex="1"),
                        ),
                        drive_selections,
                    ]
                ),
                self._drives_out,
            ]
        )

        config_panel, figs_output = self.viz_manager.compose()

        # Create optimizer tab
        opt_box = VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                self.widget_opt_solver,
                                self.widget_opt_obj_fun,
                            ]
                        ),
                        VBox(
                            [
                                self.widget_opt_max_iter,
                                self.widget_opt_tstop,
                            ]
                        ),
                    ]
                ),
                self.widget_opt_target_data,
                self.run_opt_button,
                self._opt_out,
            ]
        )

        # Tabs for left pane
        left_tab = Tab()
        left_tab.children = [
            simulation_box,
            connectivity_configuration,
            drives_options,
            opt_box,
            config_panel,
        ]
        titles = (
            "Simulation",
            "Network",
            "External drives",
            "Optimization",
            "Visualization",
        )
        for idx, title in enumerate(titles):
            left_tab.set_title(idx, title)

        self.app_layout = AppLayout(
            header=self._header,
            left_sidebar=VBox(
                [
                    VBox([left_tab], layout=self.layout["left_tab"]),
                    self._operation_buttons,
                    self._log_window,
                ],
                layout=self.layout["left_sidebar"],
            ),
            right_sidebar=figs_output,
            footer=self._simulation_status_bar,
            pane_widths=[
                self.layout["left_sidebar"].width,
                "0px",
                self.layout["visualization_window"].width,
            ],
            pane_heights=[
                self.layout["header_height"],
                self.layout["visualization_window"].height,
                self.layout["simulation_status_height"],
            ],
        )

        self._link_callbacks()

        # initialize connectivity, drives, and optimization ipywidgets
        self.load_conn_drives_opt()

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
        embed_minimal_html(file, views=[self.app_layout], title="")
        if not width:
            width = self.total_width + extra_margin
        if not height:
            height = self.total_height + extra_margin

        content = urllib.parse.quote(file.getvalue().encode("utf8"))
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
    def _simulate_upload_data(self, file_url):
        uploaded_value = _prepare_upload_file(file_url)
        self.load_data_button.set_trait("value", uploaded_value)

    def _simulate_upload_connectivity(self, file_url):
        uploaded_value = _prepare_upload_file(file_url)
        self.load_connectivity_button.set_trait("value", uploaded_value)

    def _simulate_upload_drives(self, file_url):
        uploaded_value = _prepare_upload_file(file_url)
        self.load_drives_button.set_trait("value", uploaded_value)

    def _simulate_left_tab_click(self, tab_title):
        # Get left tab group object
        left_tab = self.app_layout.left_sidebar.children[0].children[0]
        # Check that the title is in the tab group
        if tab_title in left_tab.titles:
            # Simulate the user clicking on the tab
            left_tab.selected_index = left_tab.titles.index(tab_title)
        else:
            raise ValueError("Tab title does not exist.")

    def _simulate_make_figure(
        self,
    ):
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

    def _simulate_delete_single_drive(self, idx=0):
        self.drive_accordion.selected_index = idx
        self.drive_boxes[idx].children[-1].click()

    def load_conn_drives_opt(self):
        """Add connectivity, drives, and optimization ipywidgets from params."""
        with self._log_out:
            # Add connectivity
            add_connectivity_tab(
                self.params,
                self._connectivity_out,
                self.connectivity_widgets,
                self._cell_params_out,
                self.cell_parameters_widgets,
                self.cell_layer_radio_buttons,
                self.cell_type_radio_buttons,
                self._global_gain_out,
                self.global_gain_widgets,
                self.layout,
            )

            # Add drives
            self.add_drive_tab(self.params)

            # Add optimization
            self.add_opt_tab(self.params)

            # AES where to do opt observe of drive values?

    def add_drive_widget(
        self,
        drive_type,
        location,
        prespecified_drive_name=None,
        prespecified_drive_data=None,
        prespecified_weights_ampa=None,
        prespecified_weights_nmda=None,
        prespecified_delays=None,
        prespecified_n_drive_cells=None,
        prespecified_cell_specific=None,
        render=True,
        expand_last_drive=True,
        event_seed=14,
    ):
        """Add a widget for a new drive."""

        # Check only adds 1 tonic input widget
        if drive_type == "Tonic" and not _is_valid_add_tonic_input(self.drive_widgets):
            return

        # Build drive widget objects
        name = (
            drive_type + str(len(self.drive_boxes))
            if not prespecified_drive_name
            else prespecified_drive_name
        )
        style = {"description_width": "125px"}
        prespecified_drive_data = (
            {} if not prespecified_drive_data else prespecified_drive_data
        )
        prespecified_drive_data.update({"seedcore": max(event_seed, 2)})

        drive, drive_box = _build_drive_objects(
            drive_type,
            name,
            self.widget_tstop,
            self.layout["drive_textbox"],
            style,
            location,
            prespecified_drive_data,
            prespecified_weights_ampa,
            prespecified_weights_nmda,
            prespecified_delays,
            prespecified_n_drive_cells,
            prespecified_cell_specific,
        )

        # Add delete button and assign its call-back function
        delete_button = Button(
            description="Delete",
            button_style="danger",
            icon="close",
            layout=self.layout["del_fig_btn"],
        )
        delete_button.on_click(self._delete_single_drive)
        drive_box.children += (
            HTML(value="<p> </p>"),  # Adds blank space
            delete_button,
        )

        self.drive_boxes.append(drive_box)
        self.drive_widgets.append(drive)

        if render:
            # Construct accordion object
            self.drive_accordion.children = self.drive_boxes
            self.drive_accordion.selected_index = (
                len(self.drive_boxes) - 1 if expand_last_drive else None
            )
            # Update accordion title with location
            for idx, drive in enumerate(self.drive_widgets):
                tab_name = drive["name"]
                if drive["type"] != "Tonic":
                    tab_name += f" ({drive['location']})"
                self.drive_accordion.set_title(idx, tab_name)

            self._drives_out.clear_output()
            with self._drives_out:
                display(self.drive_accordion)

    def add_drive_tab(self, params):
        net = dict_to_network(params)
        drive_specs = net.external_drives
        tonic_specs = net.external_biases

        # clear before adding drives
        self._drives_out.clear_output()
        while len(self.drive_widgets) > 0:
            self.drive_widgets.pop()
            self.drive_boxes.pop()

        drive_names = list(drive_specs.keys())
        # Add tonic biases
        if tonic_specs:
            drive_names.extend(list(tonic_specs.keys()))

        for idx, drive_name in enumerate(drive_names):  # order matters
            if "tonic" in drive_name:
                specs = dict(type="tonic", location=None)
                kwargs = dict(prespecified_drive_data=tonic_specs[drive_name])
            else:
                specs = drive_specs[drive_name]
                kwargs = dict(
                    prespecified_drive_data=specs["dynamics"],
                    prespecified_weights_ampa=specs["weights_ampa"],
                    prespecified_weights_nmda=specs["weights_nmda"],
                    prespecified_delays=specs["synaptic_delays"],
                    prespecified_n_drive_cells=specs["n_drive_cells"],
                    prespecified_cell_specific=specs["cell_specific"],
                    event_seed=specs["event_seed"],
                )

            should_render = idx == (len(drive_names) - 1)
            self.add_drive_widget(
                drive_type=specs["type"].capitalize(),
                location=specs["location"],
                prespecified_drive_name=drive_name,
                render=should_render,
                expand_last_drive=False,
                **kwargs,
            )

    def on_upload_params_change(self, change, layout, load_type):
        if len(change["owner"].value) == 0:
            return
        param_dict = change["new"][0]
        file_contents = codecs.decode(param_dict["content"], encoding="utf-8")

        with self._log_out:
            params = json.loads(file_contents)

            # update simulation settings and params
            if "tstop" in params.keys():
                self.widget_tstop.value = params["tstop"]
            if "dt" in params.keys():
                self.widget_dt.value = params["dt"]

            # init network, add drives & connectivity
            if load_type == "connectivity":
                add_connectivity_tab(
                    params,
                    self._connectivity_out,
                    self.connectivity_widgets,
                    self._cell_params_out,
                    self.cell_parameters_widgets,
                    self.cell_layer_radio_buttons,
                    self.cell_type_radio_buttons,
                    self._global_gain_out,
                    self.global_gain_widgets,
                    layout,
                )
            elif load_type == "drives":
                self.add_drive_tab(params)
                # AES TODO add self.add_opt_tab here too
            else:
                raise ValueError

            print(f"Loaded {load_type} from {param_dict['name']}")
        # Resets file counter to 0
        change["owner"].set_trait("value", ([]))
        return params

    def add_opt_tab(self, params):
        """Create/update the dynamic output of the optimization tab"""
        net = dict_to_network(params)
        drive_specs = net.external_drives
        tonic_specs = net.external_biases

        # clear before adding drives
        self._opt_out.clear_output()
        while len(self.opt_drive_widgets) > 0:
            self.opt_drive_widgets.pop()
            self.opt_drive_boxes.pop()

        drive_names = list(drive_specs.keys())
        # Add tonic biases
        if tonic_specs:
            drive_names.extend(list(tonic_specs.keys()))

        for idx, drive_name in enumerate(drive_names):  # order matters
            if "tonic" in drive_name:
                specs = dict(type="tonic", location=None)
                kwargs = dict(prespecified_drive_data=tonic_specs[drive_name])
            else:
                specs = drive_specs[drive_name]
                kwargs = dict(
                    prespecified_drive_data=specs["dynamics"],
                    prespecified_weights_ampa=specs["weights_ampa"],
                    prespecified_weights_nmda=specs["weights_nmda"],
                    prespecified_delays=specs["synaptic_delays"],
                    prespecified_n_drive_cells=specs["n_drive_cells"],
                    prespecified_cell_specific=specs["cell_specific"],
                    event_seed=specs["event_seed"],
                )

            should_render = idx == (len(drive_names) - 1)
            self.add_opt_widget(
                drive_type=specs["type"].capitalize(),
                location=specs["location"],
                prespecified_drive_name=drive_name,
                render=should_render,
                expand_last_drive=False,
                **kwargs,
            )

    def add_opt_widget(
        self,
        drive_type,
        location,
        prespecified_drive_name=None,
        prespecified_drive_data=None,
        prespecified_weights_ampa=None,
        prespecified_weights_nmda=None,
        prespecified_delays=None,
        prespecified_n_drive_cells=None,
        prespecified_cell_specific=None,
        render=True,
        expand_last_drive=True,
        event_seed=14,
    ):
        """Add a optimization widget for a new drive, including to the accordion."""

        # Check only adds 1 tonic input widget
        if drive_type == "Tonic" and not _is_valid_add_tonic_input(self.drive_widgets):
            return

        # Build drive widget objects
        name = (
            drive_type + str(len(self.drive_boxes))
            if not prespecified_drive_name
            else prespecified_drive_name
        )
        style = {"description_width": "125px"}
        prespecified_drive_data = (
            {} if not prespecified_drive_data else prespecified_drive_data
        )
        prespecified_drive_data.update({"seedcore": max(event_seed, 2)})

        # AES: TODO
        # 1. first make new evoked only accordion entry with checkboxes/constraints
        # 2. then build rest of execution pipeline with said constraints
        # 3. then realize my design was broken and have to start over again

        # AES TODO widget sizes UGH
        opt_drive_box, opt_drive_widget = _build_opt_objects(
            drive_type,
            name,
            self.widget_tstop,
            # Layout(width="270px", height="auto"),  #  self.layout["drive_widget"],
            # Layout(width="170px", height="auto"),  #  self.layout["drive_widget"],
            # Layout(width="140px", height="auto"),  #  self.layout["drive_widget"],
            # {"description_width": "125px"},  # style,
            self.layout["drive_textbox"],
            style,
            location,
            prespecified_drive_data,
            prespecified_weights_ampa,
            prespecified_weights_nmda,
            prespecified_delays,
            prespecified_n_drive_cells,
            prespecified_cell_specific,
        )

        # AES TODO after these are created, add observers

        # # Add delete button and assign its call-back function
        # delete_button = Button(
        #     description="Delete",
        #     button_style="danger",
        #     icon="close",
        #     layout=self.layout["del_fig_btn"],
        # )
        # delete_button.on_click(self._delete_single_drive)
        # opt_drive_box.children += (
        #     HTML(value="<p> </p>"),  # Adds blank space
        #     delete_button,
        # )

        self.opt_drive_boxes.append(opt_drive_box)
        self.opt_drive_widgets.append(opt_drive_widget)

        if render:
            # Construct accordion object
            self.opt_accordion.children = self.opt_drive_boxes
            self.opt_accordion.selected_index = (
                len(self.opt_drive_boxes) - 1 if expand_last_drive else None
            )
            # Update accordion title with location
            for idx, opt_drive_widget in enumerate(self.opt_drive_widgets):
                tab_name = opt_drive_widget["name"]
                if opt_drive_widget["type"] != "Tonic":
                    tab_name += f" ({opt_drive_widget['location']})"
                self.opt_accordion.set_title(idx, tab_name)

            self._opt_out.clear_output()
            with self._opt_out:
                display(self.opt_accordion)


def _prepare_upload_file_from_local(path):
    path = Path(path)
    with open(path, "rb") as file:
        content = memoryview(file.read())
    last_modified = datetime.fromtimestamp(path.stat().st_mtime)

    upload_structure = [
        {
            "name": path.name,
            "type": mimetypes.guess_type(path)[0],
            "size": path.stat().st_size,
            "content": content,
            "last_modified": last_modified,
        }
    ]

    return upload_structure


def _prepare_upload_file_from_url(file_url):
    file_name = file_url.split("/")[-1]
    data = urllib.request.urlopen(file_url)
    content = bytearray()
    for line in data:
        content.extend(line)

    upload_structure = [
        {
            "name": file_name,
            "type": mimetypes.guess_type(file_url)[0],
            "size": len(content),
            "content": memoryview(content),
            "last_modified": datetime.now(),
        }
    ]

    return upload_structure


def _prepare_upload_file(path):
    """Simulates output of the FileUpload widget for testing.

    Unit tests for the GUI simulate user upload of files. File source can
    either be local or from a URL. This function returns the data structure
    of the ipywidget FileUpload widget, a list of dictionaries with file
    attributes.
    """
    try:
        uploaded_value = _prepare_upload_file_from_local(path)
    except (FileNotFoundError, OSError):
        uploaded_value = _prepare_upload_file_from_url(path)

    return uploaded_value


def _update_nested_dict(original, new, skip_none=True):
    """Updates dictionary values from another dictionary

    Will update nested dictionaries in the structure. New items from the
    update dictionary are added and omitted items are retained from the
    original dictionary. By default, will not pass None values from the update
    dictionary.

    Parameters
    ----------
    original : dict
        Dictionary to update
    new : dict
        Dictionary with new values for updating
    skip_none : bool, default True
        None values in the new dictionary are not passed to the updated
        dictionary by when True. If False None values will be passed to the
        updated dictionary.

    Returns dict
    -------

    """
    updated = original.copy()
    for key, value in new.items():
        if (
            isinstance(value, dict)
            and key in updated
            and isinstance(updated[key], dict)
        ):
            updated[key] = _update_nested_dict(updated[key], value, skip_none)
        elif (value is not None) or (not skip_none):
            updated[key] = value
        else:
            pass

    return updated


def create_expanded_button(
    description, button_style, layout, disabled=False, button_color="#8A2BE2"
):
    return Button(
        description=description,
        button_style=button_style,
        layout=layout,
        style={"button_color": button_color},
        disabled=disabled,
    )


def _get_connectivity_widgets(conn_data, global_gain_textfields):
    """Create connectivity box widgets from specified weight and gains"""
    style = {"description_width": "100px"}
    html_tab = "&emsp;"

    sliders = list()
    for receptor_idx, receptor_name in enumerate(conn_data.keys()):
        global_gain_type = global_gain_type_lookup_dict[
            (
                conn_data[receptor_name]["src_gids"],
                conn_data[receptor_name]["target_gids"],
            )
        ]

        weight_text_input = BoundedFloatText(
            value=conn_data[receptor_name]["weight"],
            disabled=False,
            continuous_update=False,
            min=0,
            max=1e6,
            step=0.01,
            description="Weight:",
            style=style,
        )

        single_gain_text_input = BoundedFloatText(
            value=conn_data[receptor_name]["gain"],
            disabled=False,
            continuous_update=False,
            min=0,
            max=1e6,
            step=0.1,
            description="Gain:",
            style=style,
        )

        combined_gain_indicator_output = HTML(
            value=f"""
            <p style='margin:0px;padding-left:115px;'>
            <b>Total Computed Gain={
                (
                    1
                    + (global_gain_textfields[global_gain_type].value - 1)
                    + (single_gain_text_input.value - 1)
                ):.2f}</b></p>"""
        )

        # Create closure to capture current widget references
        def make_update_gain_indicator(gain_output, gain_input, gain_type):
            def update_gain_indicator(change):
                gain_output.value = f"""
                <p style='margin:0px;padding-left:115px;'>
                <b>Total Computed Gain={
                    (
                        1
                        + (global_gain_textfields[gain_type].value - 1)
                        + (gain_input.value - 1)
                    ):.2f}</b></p>"""

            return update_gain_indicator

        # Connect the gain objects so they update each other
        update_gain_indicator_fn = make_update_gain_indicator(
            combined_gain_indicator_output, single_gain_text_input, global_gain_type
        )
        global_gain_textfields[global_gain_type].observe(
            update_gain_indicator_fn, names="value"
        )
        single_gain_text_input.observe(update_gain_indicator_fn, names="value")

        # Now, create the similar final Weight text output
        final_weight_indicator_output = HTML(
            value=f"""
            <b>Final Weight={
                (
                    weight_text_input.value
                    * (
                        1
                        + (global_gain_textfields[global_gain_type].value - 1)
                        + (single_gain_text_input.value - 1)
                    )
                ):.4f}</b>"""
        )

        # Create closure to capture current widget references
        def make_update_weight_indicator(
            weight_output, weight_input, gain_input, gain_type
        ):
            def update_weight_indicator(change):
                weight_output.value = f"""
                <b>Final Weight={
                    (
                        weight_input.value
                        * (
                            1
                            + (global_gain_textfields[gain_type].value - 1)
                            + (gain_input.value - 1)
                        )
                    ):.4f}</b>"""

            return update_weight_indicator

        # Connect the weight and gain objects so they update each other
        update_weight_indicator_fn = make_update_weight_indicator(
            final_weight_indicator_output,
            weight_text_input,
            single_gain_text_input,
            global_gain_type,
        )
        global_gain_textfields[global_gain_type].observe(
            update_weight_indicator_fn, names="value"
        )
        single_gain_text_input.observe(update_weight_indicator_fn, names="value")
        weight_text_input.observe(update_weight_indicator_fn, names="value")

        display_name = conn_data[receptor_name]["receptor"].upper()

        map_display_names = {
            "GABAA": "GABA<sub>A</sub>",
            "GABAB": "GABA<sub>B</sub>",
        }

        if display_name in map_display_names:
            display_name = map_display_names[display_name]

        conn_widget = VBox(
            [
                HTML(
                    value=f"""<p style='margin:5px;'><b>{html_tab}{html_tab}
            Receptor: {display_name}</b></p>"""
                ),
                HBox(
                    [
                        weight_text_input,
                        final_weight_indicator_output,
                    ]
                ),
                HBox(
                    [
                        single_gain_text_input,
                    ]
                ),
                combined_gain_indicator_output,
            ]
        )

        #  Add class to child Vboxes for targeted CSS
        conn_widget.add_class("connectivity-subsection")
        conn_widget._belongsto = {
            "receptor": conn_data[receptor_name]["receptor"],
            "location": conn_data[receptor_name]["location"],
            "src_gids": conn_data[receptor_name]["src_gids"],
            "target_gids": conn_data[receptor_name]["target_gids"],
        }
        sliders.append(conn_widget)

    return sliders


def _get_drive_weight_widgets(layout, style, location, data=None):
    default_data = {
        "weights_ampa": {
            "L5_pyramidal": 0.0,
            "L2_pyramidal": 0.0,
            "L5_basket": 0.0,
            "L2_basket": 0.0,
        },
        "weights_nmda": {
            "L5_pyramidal": 0.0,
            "L2_pyramidal": 0.0,
            "L5_basket": 0.0,
            "L2_basket": 0.0,
        },
        "delays": {
            "L5_pyramidal": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 0.1,
            "L2_basket": 0.1,
        },
    }
    if isinstance(data, dict):
        default_data = _update_nested_dict(default_data, data)

    kwargs = dict(layout=layout, style=style)
    cell_types = ["L5_pyramidal", "L2_pyramidal", "L5_basket", "L2_basket"]
    if location == "distal":
        cell_types.remove("L5_basket")

    weights_ampa, weights_nmda, delays = dict(), dict(), dict()
    for cell_type in cell_types:
        weights_ampa[f"{cell_type}"] = BoundedFloatText(
            value=default_data["weights_ampa"][cell_type],
            description=f"{cell_type}:",
            min=0,
            max=1e6,
            step=0.01,
            **kwargs,
        )
        weights_nmda[f"{cell_type}"] = BoundedFloatText(
            value=default_data["weights_nmda"][cell_type],
            description=f"{cell_type}:",
            min=0,
            max=1e6,
            step=0.01,
            **kwargs,
        )
        delays[f"{cell_type}"] = BoundedFloatText(
            value=default_data["delays"][cell_type],
            description=f"{cell_type}:",
            min=0,
            max=1e6,
            step=0.1,
            **kwargs,
        )

    widgets_dict = {
        "weights_ampa": weights_ampa,
        "weights_nmda": weights_nmda,
        "delays": delays,
    }
    widgets_list = (
        [HTML(value="<b>AMPA weights</b>")]
        + list(weights_ampa.values())
        + [HTML(value="<b>NMDA weights</b>")]
        + list(weights_nmda.values())
        + [HTML(value="<b>Synaptic delays</b>")]
        + list(delays.values())
    )
    return widgets_list, widgets_dict


def _cell_spec_change(change, widget):
    if change["new"]:
        widget.disabled = True
    else:
        widget.disabled = False


def _get_rhythmic_widget_for_drives(
    name,
    tstop_widget,
    layout,
    style,
    location,
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
):
    default_data = {
        "tstart": 0.0,
        "tstart_std": 0.0,
        "tstop": tstop_widget.value,
        "burst_rate": 7.5,
        "burst_std": 0,
        "numspikes": 1,
        "n_drive_cells": 1,
        "cell_specific": False,
        "seedcore": 14,
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    default_data = _update_nested_dict(default_data, data)

    kwargs = dict(layout=layout, style=style)
    tstart = BoundedFloatText(
        value=default_data["tstart"],
        description="Start time (ms)",
        min=0,
        max=1e6,
        **kwargs,
    )
    tstart_std = BoundedFloatText(
        value=default_data["tstart_std"],
        description="Start time dev (ms)",
        min=0,
        max=1e6,
        **kwargs,
    )
    tstop = BoundedFloatText(
        value=default_data["tstop"],
        description="Stop time (ms)",
        max=tstop_widget.value,
        **kwargs,
    )
    burst_rate = BoundedFloatText(
        value=default_data["burst_rate"],
        description="Burst rate (Hz)",
        min=0,
        max=1e6,
        **kwargs,
    )
    burst_std = BoundedFloatText(
        value=default_data["burst_std"],
        description="Burst std dev (Hz)",
        min=0,
        max=1e6,
        **kwargs,
    )
    numspikes = BoundedIntText(
        value=default_data["numspikes"],
        description="No. Spikes:",
        min=0,
        max=int(1e6),
        **kwargs,
    )
    n_drive_cells = IntText(
        value=default_data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=default_data["cell_specific"],
        **kwargs,
    )
    cell_specific = Checkbox(
        value=default_data["cell_specific"], description="Cell-Specific", **kwargs
    )
    seedcore = IntText(value=default_data["seedcore"], description="Seed", **kwargs)

    widgets_list, widgets_dict = _get_drive_weight_widgets(
        layout,
        style,
        location,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
    )

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    drive_box = VBox(
        [
            tstart,
            tstart_std,
            tstop,
            burst_rate,
            burst_std,
            numspikes,
            n_drive_cells,
            cell_specific,
            seedcore,
        ]
        + widgets_list
    )

    drive = dict(
        type="Rhythmic",
        name=name,
        tstart=tstart,
        tstart_std=tstart_std,
        burst_rate=burst_rate,
        burst_std=burst_std,
        numspikes=numspikes,
        seedcore=seedcore,
        location=location,
        tstop=tstop,
        n_drive_cells=n_drive_cells,
        is_cell_specific=cell_specific,
    )
    drive.update(widgets_dict)
    return drive, drive_box


def _get_poisson_widget_for_drives(
    name,
    tstop_widget,
    layout,
    style,
    location,
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
):
    default_data = {
        "tstart": 0.0,
        "tstop": tstop_widget.value,
        "n_drive_cells": 1,
        "cell_specific": True,
        "seedcore": 14,
        "rate_constant": {
            "L2_pyramidal": 40.0,
            "L5_pyramidal": 40.0,
            "L2_basket": 40.0,
            "L5_basket": 40.0,
        },
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    default_data = _update_nested_dict(default_data, data)

    tstart = BoundedFloatText(
        value=default_data["tstart"],
        description="Start time (ms)",
        min=0,
        max=1e6,
        layout=layout,
        style=style,
    )
    tstop = BoundedFloatText(
        value=default_data["tstop"],
        max=tstop_widget.value,
        description="Stop time (ms)",
        layout=layout,
        style=style,
    )
    n_drive_cells = IntText(
        value=default_data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=default_data["cell_specific"],
        layout=layout,
        style=style,
    )
    cell_specific = Checkbox(
        value=default_data["cell_specific"],
        description="Cell-Specific",
        layout=layout,
        style=style,
    )
    seedcore = IntText(
        value=default_data["seedcore"], description="Seed", layout=layout, style=style
    )

    cell_types = ["L5_pyramidal", "L2_pyramidal", "L5_basket", "L2_basket"]
    rate_constant = dict()
    for cell_type in cell_types:
        rate_constant[f"{cell_type}"] = BoundedFloatText(
            value=default_data["rate_constant"][cell_type],
            description=f"{cell_type}:",
            min=0,
            max=1e6,
            step=0.01,
            layout=layout,
            style=style,
        )

    widgets_list, widgets_dict = _get_drive_weight_widgets(
        layout,
        style,
        location,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
    )
    widgets_dict.update({"rate_constant": rate_constant})
    widgets_list.extend(
        [HTML(value="<b>Rate constants</b>")]
        + list(widgets_dict["rate_constant"].values())
    )

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    drive_box = VBox(
        [tstart, tstop, n_drive_cells, cell_specific, seedcore] + widgets_list
    )
    drive = dict(
        type="Poisson",
        name=name,
        tstart=tstart,
        tstop=tstop,
        rate_constant=rate_constant,
        seedcore=seedcore,
        location=location,  # notice this is a widget but a str!
        n_drive_cells=n_drive_cells,
        is_cell_specific=cell_specific,
    )
    drive.update(widgets_dict)
    return drive, drive_box


def _get_evoked_widget_for_drives(
    name,
    layout,
    style,
    location,
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
):
    default_data = {
        "mu": 0,
        "sigma": 1,
        "numspikes": 1,
        "n_drive_cells": 1,
        "cell_specific": True,
        "seedcore": 14,
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    default_data = _update_nested_dict(default_data, data)

    kwargs = dict(layout=layout, style=style)
    mu = BoundedFloatText(
        value=default_data["mu"],
        description="Mean time:",
        min=0,
        max=1e6,
        step=0.01,
        **kwargs,
    )
    sigma = BoundedFloatText(
        value=default_data["sigma"],
        description="Std dev time:",
        min=0,
        max=1e6,
        step=0.01,
        **kwargs,
    )
    numspikes = IntText(
        value=default_data["numspikes"], description="No. Spikes:", **kwargs
    )
    n_drive_cells = IntText(
        value=default_data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=default_data["cell_specific"],
        **kwargs,
    )
    cell_specific = Checkbox(
        value=default_data["cell_specific"], description="Cell-Specific", **kwargs
    )
    seedcore = IntText(value=default_data["seedcore"], description="Seed: ", **kwargs)

    widgets_list, widgets_dict = _get_drive_weight_widgets(
        layout,
        style,
        location,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
    )

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    drive_box = VBox(
        [
            mu,
            sigma,
            numspikes,
            n_drive_cells,
            cell_specific,
            seedcore,
        ]
        + widgets_list
    )
    drive = dict(
        type="Evoked",
        name=name,
        mu=mu,
        sigma=sigma,
        numspikes=numspikes,
        seedcore=seedcore,
        location=location,
        sync_within_trial=False,
        n_drive_cells=n_drive_cells,
        is_cell_specific=cell_specific,
    )
    drive.update(widgets_dict)
    return drive, drive_box


def _get_tonic_widget_for_drives(name, tstop_widget, layout, style, data=None):
    cell_types = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]
    default_values = {"amplitude": 0, "t0": 0, "tstop": tstop_widget.value}
    t0 = default_values["t0"]
    tstop = default_values["tstop"]
    default_data = {cell_type: default_values for cell_type in cell_types}

    kwargs = dict(layout=layout, style=style)
    if isinstance(data, dict):
        default_data = _update_nested_dict(default_data, data)

    amplitudes = dict()
    for cell_type in cell_types:
        amplitude = default_data[cell_type]["amplitude"]
        amplitudes[cell_type] = BoundedFloatText(
            value=amplitude, description=cell_type, min=0, max=1e6, step=0.01, **kwargs
        )
        # Reset the global t0 and stop with values from the 'data' keyword.
        # It should be same across all the cell-types.
        if amplitude > 0:
            t0 = default_data[cell_type]["t0"]
            tstop = default_data[cell_type]["tstop"]

    start_times = BoundedFloatText(
        value=t0, description="Start time", min=0, max=1e6, step=1.0, **kwargs
    )
    stop_times = BoundedFloatText(
        value=tstop, description="Stop time", min=-1, max=1e6, step=1.0, **kwargs
    )

    widgets_dict = {"amplitude": amplitudes, "t0": start_times, "tstop": stop_times}
    widgets_list = (
        [HTML(value="<b>Times (ms):</b>")]
        + [start_times, stop_times]
        + [HTML(value="<b>Amplitude (nA):</b>")]
        + list(amplitudes.values())
    )

    drive_box = VBox(widgets_list)
    drive = dict(
        type="Tonic",
        name=name,
        amplitude=amplitudes,
        t0=start_times,
        tstop=stop_times,
    )

    drive.update(widgets_dict)
    return drive, drive_box


def _build_drive_objects(
    drive_type,
    name,
    tstop_widget,
    layout,
    style,
    location,
    drive_data,
    weights_ampa,
    weights_nmda,
    delays,
    n_drive_cells,
    cell_specific,
):
    if drive_type in ("Rhythmic", "Bursty"):
        drive, drive_box = _get_rhythmic_widget_for_drives(
            name,
            tstop_widget,
            layout,
            style,
            location,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
        )
    elif drive_type == "Poisson":
        drive, drive_box = _get_poisson_widget_for_drives(
            name,
            tstop_widget,
            layout,
            style,
            location,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
        )
    elif drive_type in ("Evoked", "Gaussian"):
        drive, drive_box = _get_evoked_widget_for_drives(
            name,
            layout,
            style,
            location,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
        )
    elif drive_type == "Tonic":
        drive, drive_box = _get_tonic_widget_for_drives(
            name, tstop_widget, layout, style, data=drive_data
        )
    else:
        raise ValueError(f"Unknown drive type {drive_type}")

    return drive, drive_box


def add_connectivity_tab(
    params,
    connectivity_out,
    connectivity_textfields,
    cell_params_out,
    cell_parameters_vboxes,
    cell_layer_radio_button,
    cell_type_radio_button,
    global_gain_out,
    global_gain_textfields,
    layout,
):
    """Add all possible connectivity boxes to connectivity tab."""
    net = dict_to_network(params)

    # build network connectivity tab
    add_network_connectivity_tab(
        net,
        connectivity_out,
        connectivity_textfields,
        global_gain_out,
        global_gain_textfields,
        layout,
    )

    # build cell parameters tab
    add_cell_parameters_tab(
        cell_params_out,
        cell_parameters_vboxes,
        cell_layer_radio_button,
        cell_type_radio_button,
        layout,
    )

    return net


def add_network_connectivity_tab(
    net,
    connectivity_out,
    connectivity_textfields,
    global_gain_out,
    global_gain_textfields,
    layout,
):
    """Creates widgets for synaptic connectivity values and global synaptic gains"""

    ### Global synaptic gains
    # ---------------------------------------------------------------
    global_gain_out.clear_output()
    gain_values = net.get_global_synaptic_gains()
    gain_types = ("e_e", "e_i", "i_e", "i_i")

    # Same as _get_connectivity_widgets
    style = {"description_width": "100px"}

    for gain_type in gain_types:
        gain_widget = BoundedFloatText(
            value=gain_values[gain_type],
            description=f"{global_gain_type_display_dict[gain_type]}",
            min=0,
            max=1e6,
            step=0.1,
            disabled=False,
            style=style,
        )

        gain_widget.layout.width = "220px"
        global_gain_textfields[gain_type] = gain_widget

    title_box = HTML(
        """
        <div
        style="
            background: gray;
            color: white;
            width: 100%;
            margin-bottom: 2px;
            text-align: center;
        ">
        Global Synaptic Gain Modifiers
        </div>
        """
    )

    left_box = VBox(
        [
            global_gain_textfields["e_e"],
            global_gain_textfields["e_i"],
        ]
    )
    right_box = VBox(
        [
            global_gain_textfields["i_e"],
            global_gain_textfields["i_i"],
        ]
    )
    gain_vbox = VBox(
        [
            title_box,
            HBox(
                [
                    left_box,
                    right_box,
                ]
            ),
        ],
        layout=Layout(
            border="1px solid #bdbdbd",  # border width, style, and color
            padding="10px",  # optional: adds space inside the border
        ),
    )

    with global_gain_out:
        display(gain_vbox)

    ### Connectivity accordion
    # ---------------------------------------------------------------
    cell_types = [ct for ct in net.cell_types.keys()]
    receptors = ("ampa", "nmda", "gabaa", "gabab")
    locations = ("proximal", "distal", "soma")

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
                    conn_indices = pick_connection(
                        net=net,
                        src_gids=src_gids,
                        target_gids=target_gids,
                        loc=location,
                        receptor=receptor,
                    )
                    if len(conn_indices) > 0:
                        assert len(conn_indices) == 1
                        conn_idx = conn_indices[0]
                        current_w = net.connectivity[conn_idx]["nc_dict"]["A_weight"]
                        current_p = net.connectivity[conn_idx]["probability"]
                        current_g = net.connectivity[conn_idx]["nc_dict"]["gain"]
                        # valid connection
                        receptor_related_conn[receptor] = {
                            "weight": current_w,
                            "probability": current_p,
                            "gain": current_g,
                            # info used to identify connection
                            "receptor": receptor,
                            "location": location,
                            "src_gids": src_gids,
                            "target_gids": target_gids,
                        }
                if len(receptor_related_conn) > 0:
                    connectivity_names.append(f"{src_gids}→{target_gids} ({location})")
                    connectivity_textfields.append(
                        _get_connectivity_widgets(
                            receptor_related_conn, global_gain_textfields
                        )
                    )

    # Style the contents of the Connectivity Tab
    # -------------------------------------------------------------------------

    # define custom Vbox layout
    # no_padding_layout = Layout(padding="0", margin="0") # unused

    # Initialize sections within the Accordion

    connectivity_boxes = [VBox(slider) for slider in connectivity_textfields]

    # Add class to child Vboxes for targeted CSS
    for box in connectivity_boxes:
        box.add_class("connectivity-contents")

    # Initialize the Accordion section

    cell_connectivity = Accordion(children=connectivity_boxes)

    # Add class to Accordion section for targeted CSS
    cell_connectivity.add_class("connectivity-section")

    for idx, connectivity_name in enumerate(connectivity_names):
        cell_connectivity.set_title(idx, connectivity_name)

    # Style the <div> automatically created around connectivity boxes
    connectivity_out_style = HTML("""
    <style>
        /* CSS to style elements inside the Accordion */
        .connectivity-section .jupyter-widget-Collapse-contents {
            padding: 0px 0px 10px 0px !important;
            margin: 0 !important;
        }
    </style>
    """)

    # Display the Accordion with styling
    with connectivity_out:
        display(connectivity_out_style, cell_connectivity)

    return net


def add_cell_parameters_tab(
    cell_params_out,
    cell_parameters_vboxes,
    cell_layer_radio_button,
    cell_type_radio_button,
    layout,
):
    L2_default_values = get_L2Pyr_params_default()
    L5_default_values = get_L5Pyr_params_default()
    cell_types = [("L2", L2_default_values), ("L5", L5_default_values)]
    style = {"description_width": "255px"}
    kwargs = dict(layout=layout, style=style)

    for cell_type in cell_types:
        layer_parameters = list()
        for layer in cell_parameters_dict.keys():
            if ("Biophysic" in layer or "Geometry" in layer) and cell_type[
                0
            ] not in layer:
                continue

            for parameter in cell_parameters_dict[layer]:
                param_name, param_units, params_key = (
                    parameter[0],
                    parameter[1],
                    parameter[2],
                )
                default_value = get_cell_param_default_value(
                    f"{cell_type[0]}Pyr_{params_key}", cell_type[1]
                )
                description = f"{param_name} ({param_units})"
                min_value = -1000.0 if param_units not in "ms" else 0
                text_field = BoundedFloatText(
                    value=default_value,
                    min=min_value,
                    max=1000.0,
                    step=0.1,
                    description=description,
                    disabled=False,
                    **kwargs,
                )
                text_field.layout.width = "350px"
                layer_parameters.append(text_field)
            cell_parameters_key = f"{cell_type[0]} Pyramidal_{layer}"
            cell_parameters_vboxes[cell_parameters_key] = VBox(layer_parameters)
            layer_parameters.clear()

    # clear existing connectivity
    cell_params_out.clear_output()

    # Add cell parameters
    _update_cell_params_vbox(
        cell_params_out,
        cell_parameters_vboxes,
        cell_type_radio_button.value,
        cell_layer_radio_button.value,
    )


def get_cell_param_default_value(cell_type_key, param_dict):
    return param_dict[cell_type_key]


def on_upload_data_change(change, data, viz_manager, log_out):
    if len(change["owner"].value) == 0:
        return
    # Parsing file information from the 'change' object passed in from
    # the upload file widget.
    data_dict = change["new"][0]
    dict_name = data_dict["name"].rsplit(".", 1)
    data_fname = dict_name[0]
    file_extension = f".{dict_name[1]}"

    # If data was already loaded return
    if data_fname in data["simulation_data"].keys():
        with log_out:
            logger.error(f"Found existing data: {data_fname}.")
        return

    # Read the file
    ext_content = data_dict["content"]
    ext_content = codecs.decode(ext_content, encoding="utf-8")
    with log_out:
        # Write loaded data to data object
        data["simulation_data"][data_fname] = {
            "net": None,
            "dpls": [_read_dipole_txt(io.StringIO(ext_content), file_extension)],
        }
        logger.info(f"External data {data_fname} loaded.")

        # Create a dipole plot
        _template_name = "[Blank] single figure"
        viz_manager.reset_fig_config_tabs(template_name=_template_name)
        viz_manager.add_figure()
        fig_name = _idx2figname(viz_manager.data["fig_idx"]["idx"] - 1)
        process_configs = {"dipole_smooth": 0, "dipole_scaling": 1}
        viz_manager._simulate_edit_figure(
            fig_name,
            ax_name="ax0",
            simulation_name=data_fname,
            plot_type="current dipole",
            preprocessing_config=process_configs,
            operation="plot",
        )
        # Reset the load file widget
        change["owner"].value = []


def _drive_widget_to_dict(drive, name):
    """Creates a dict of input widget values

    Input widgets for drive parameters are structured in a nested
    dictionary. This function recreates the nested dictionary replacing
    the input widget with its stored value.
    Parameters
    ----------
    drive : dict
        The drive dictionary containing nested dictionaries for parameters with
        multiple input widgets.
    name : str
        key of the nested dictionary

    Returns : dict
    -------

    """
    return {k: v.value for k, v in drive[name].items()}


# AES tstop appears to be unused
def _init_network_from_widgets(
    params,
    dt,
    tstop,
    single_simulation_data,
    drive_widgets,
    connectivity_textfields,
    cell_params_vboxes,
    global_gain_textfields,
    add_drive=True,
):
    """Construct network and add drives."""
    print("init network")
    single_simulation_data["net"] = dict_to_network(
        params, read_drives=False, read_external_biases=False
    )

    # Update with synaptic gains
    global_gain_values = {
        key: widget.value for key, widget in global_gain_textfields.items()
    }

    # adjust connectivity according to the connectivity_tab
    for connectivity_slider in connectivity_textfields:
        for vbox_key in connectivity_slider:
            conn_indices = pick_connection(
                net=single_simulation_data["net"],
                src_gids=vbox_key._belongsto["src_gids"],
                target_gids=vbox_key._belongsto["target_gids"],
                loc=vbox_key._belongsto["location"],
                receptor=vbox_key._belongsto["receptor"],
            )

            if len(conn_indices) > 0:
                assert len(conn_indices) == 1
                conn_idx = conn_indices[0]
                single_simulation_data["net"].connectivity[conn_idx]["nc_dict"][
                    "A_weight"
                ] = vbox_key.children[1].children[0].value

                # 1. identify which case of global_gain_textfield applies to this src/target
                global_gain_type = global_gain_type_lookup_dict[
                    (
                        vbox_key._belongsto["src_gids"],
                        vbox_key._belongsto["target_gids"],
                    )
                ]
                applied_global_gain_value = global_gain_values[global_gain_type]

                # 2. Multiply global by single synapse gain to get total
                single_simulation_data["net"].connectivity[conn_idx]["nc_dict"][
                    "gain"
                ] = (
                    1
                    + (applied_global_gain_value - 1)
                    + (vbox_key.children[2].children[0].value - 1)
                )

    # Update cell params
    update_functions = {
        "L2 Geometry": _update_L2_geometry_cell_params,
        "L5 Geometry": _update_L5_geometry_cell_params,
        "Synapses": _update_synapse_cell_params,
        "L2 Pyramidal_Biophysics": _update_L2_biophysics_cell_params,
        "L5 Pyramidal_Biophysics": _update_L5_biophysics_cell_params,
    }

    # Update cell params
    for vbox_key, cell_param_list in cell_params_vboxes.items():
        for key, update_function in update_functions.items():
            if key in vbox_key:
                cell_type = vbox_key.split()[0]
                update_function(
                    single_simulation_data["net"], cell_type, cell_param_list.children
                )
                break  # update needed only once per vbox_key

    for cell_type in single_simulation_data["net"].cell_types.keys():
        single_simulation_data["net"].cell_types[cell_type][
            "cell_object"
        ]._update_end_pts()
        single_simulation_data["net"].cell_types[cell_type][
            "cell_object"
        ]._compute_section_mechs()

    if add_drive is False:
        return
    # add drives to network
    for drive in drive_widgets:
        if drive["type"] in ("Tonic"):
            weights_amplitudes = _drive_widget_to_dict(drive, "amplitude")
            single_simulation_data["net"].add_tonic_bias(
                amplitude=weights_amplitudes,
                t0=drive["t0"].value,
                tstop=drive["tstop"].value,
            )
        else:
            sync_inputs_kwargs = dict(
                n_drive_cells=(
                    "n_cells"
                    if drive["is_cell_specific"].value
                    else drive["n_drive_cells"].value
                ),
                cell_specific=drive["is_cell_specific"].value,
            )

            weights_ampa = _drive_widget_to_dict(drive, "weights_ampa")
            weights_nmda = _drive_widget_to_dict(drive, "weights_nmda")
            synaptic_delays = _drive_widget_to_dict(drive, "delays")
            print(f"drive type is {drive['type']}, location={drive['location']}")
            if drive["type"] == "Poisson":
                rate_constant = _drive_widget_to_dict(drive, "rate_constant")

                single_simulation_data["net"].add_poisson_drive(
                    name=drive["name"],
                    tstart=drive["tstart"].value,
                    tstop=drive["tstop"].value,
                    rate_constant=rate_constant,
                    location=drive["location"],
                    weights_ampa=weights_ampa,
                    weights_nmda=weights_nmda,
                    synaptic_delays=synaptic_delays,
                    space_constant=100.0,
                    event_seed=drive["seedcore"].value,
                    **sync_inputs_kwargs,
                )
            elif drive["type"] in ("Evoked", "Gaussian"):
                single_simulation_data["net"].add_evoked_drive(
                    name=drive["name"],
                    mu=drive["mu"].value,
                    sigma=drive["sigma"].value,
                    numspikes=drive["numspikes"].value,
                    location=drive["location"],
                    weights_ampa=weights_ampa,
                    weights_nmda=weights_nmda,
                    synaptic_delays=synaptic_delays,
                    space_constant=3.0,
                    event_seed=drive["seedcore"].value,
                    **sync_inputs_kwargs,
                )
            elif drive["type"] in ("Rhythmic", "Bursty"):
                single_simulation_data["net"].add_bursty_drive(
                    name=drive["name"],
                    tstart=drive["tstart"].value,
                    tstart_std=drive["tstart_std"].value,
                    tstop=drive["tstop"].value,
                    location=drive["location"],
                    burst_rate=drive["burst_rate"].value,
                    burst_std=drive["burst_std"].value,
                    numspikes=drive["numspikes"].value,
                    weights_ampa=weights_ampa,
                    weights_nmda=weights_nmda,
                    synaptic_delays=synaptic_delays,
                    event_seed=drive["seedcore"].value,
                    **sync_inputs_kwargs,
                )


def run_button_clicked(
    widget_simulation_name,
    log_out,
    drive_widgets,
    all_data,
    dt,
    tstop,
    fig_default_params,
    widget_default_smoothing,
    widget_default_scaling,
    widget_min_frequency,
    widget_max_frequency,
    ntrials,
    backend_selection,
    mpi_cmd,
    n_jobs,
    params,
    simulation_status_bar,
    simulation_status_contents,
    connectivity_textfields,
    viz_manager,
    simulations_list_widget,
    cell_parameters_widgets,
    global_gain_textfields,
):
    """Run the simulation and plot outputs."""
    simulation_data = all_data["simulation_data"]
    with log_out:
        # clear empty trash simulations
        for _name in tuple(simulation_data.keys()):
            if len(simulation_data[_name]["dpls"]) == 0:
                del simulation_data[_name]

        _sim_name = widget_simulation_name.value
        if simulation_data[_sim_name]["net"] is not None:
            print("Simulation with the same name exists!")
            simulation_status_bar.value = simulation_status_contents["failed"]
            return

        _init_network_from_widgets(
            params,
            dt,
            tstop,
            simulation_data[_sim_name],
            drive_widgets,
            connectivity_textfields,
            cell_parameters_widgets,
            global_gain_textfields,
        )

        print("start simulation")
        if backend_selection.value == "MPI":
            # 'use_hwthreading_if_found' and 'sensible_default_cores' have
            # already been set elsewhere, and do not need to be re-set here.
            # Hardware-threading and oversubscription will always be disabled
            # to prevent edge cases in the GUI.
            backend = MPIBackend(
                n_procs=n_jobs.value,
                mpi_cmd=mpi_cmd.value,
                override_hwthreading_option=False,
                override_oversubscribe_option=False,
            )
        else:
            backend = JoblibBackend(n_jobs=n_jobs.value)
            print(f"Using Joblib with {n_jobs.value} core(s).")
        with backend:
            simulation_status_bar.value = simulation_status_contents["running"]
            simulation_data[_sim_name]["dpls"] = simulate_dipole(
                simulation_data[_sim_name]["net"],
                tstop=tstop.value,
                dt=dt.value,
                n_trials=ntrials.value,
            )

            simulation_status_bar.value = simulation_status_contents["finished"]

            sim_names = [
                sim_name
                for sim_name in simulation_data
                if simulation_data[sim_name]["net"] is not None
            ]

            simulations_list_widget.options = sim_names
            simulations_list_widget.value = sim_names[0]

    viz_manager.reset_fig_config_tabs()

    # update default visualization params in gui based on widget
    fig_default_params["default_smoothing"] = widget_default_smoothing.value
    fig_default_params["default_scaling"] = widget_default_scaling.value
    fig_default_params["default_min_frequency"] = widget_min_frequency.value
    fig_default_params["default_max_frequency"] = widget_max_frequency.value

    # change default visualization params in viz_manager to mirror gui
    for widget, value in fig_default_params.items():
        viz_manager.fig_default_params[widget] = value

    viz_manager.add_figure()
    fig_name = _idx2figname(viz_manager.data["fig_idx"]["idx"] - 1)
    ax_plots = [("ax0", "input histogram"), ("ax1", "current dipole")]
    for ax_name, plot_type in ax_plots:
        viz_manager._simulate_edit_figure(
            fig_name, ax_name, _sim_name, plot_type, {}, "plot"
        )


def _update_cell_params_vbox(
    cell_type_out, cell_parameters_list, cell_type, cell_layer
):
    cell_parameters_key = f"{cell_type}_{cell_layer}"
    if cell_layer in ["Biophysics", "Geometry"]:
        cell_parameters_key += f" {cell_type.split(' ')[0]}"

    # Needed for the button to display L2/3, but the underlying data to use L2
    cell_parameters_key = cell_parameters_key.replace("L2/3", "L2")

    if cell_parameters_key in cell_parameters_list:
        cell_type_out.clear_output()
        with cell_type_out:
            display(cell_parameters_list[cell_parameters_key])


def _update_L2_geometry_cell_params(net, cell_param_key, param_list):
    cell_params = param_list
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"

    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    sections["soma"]._L = cell_params[0].value
    sections["soma"]._diam = cell_params[1].value
    sections["soma"]._cm = cell_params[2].value
    sections["soma"]._Ra = cell_params[3].value

    # Dendrite common parameters
    dendrite_cm = cell_params[4].value
    dendrite_Ra = cell_params[5].value

    dendrite_sections = [name for name in sections.keys() if name != "soma"]

    param_indices = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]

    # Dendrite
    for section, indices in zip(dendrite_sections, param_indices):
        sections[section]._L = cell_params[indices[0]].value
        sections[section]._diam = cell_params[indices[1]].value
        sections[section]._cm = dendrite_cm
        sections[section]._Ra = dendrite_Ra


def _update_L5_geometry_cell_params(net, cell_param_key, param_list):
    cell_params = param_list
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"

    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    sections["soma"]._L = cell_params[0].value
    sections["soma"]._diam = cell_params[1].value
    sections["soma"]._cm = cell_params[2].value
    sections["soma"]._Ra = cell_params[3].value

    # Dendrite common parameters
    dendrite_cm = cell_params[4].value
    dendrite_Ra = cell_params[5].value

    dendrite_sections = [name for name in sections.keys() if name != "soma"]

    param_indices = [
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 17),
        (18, 19),
        (20, 21),
    ]

    # Dentrite
    for section, indices in zip(dendrite_sections, param_indices):
        sections[section]._L = cell_params[indices[0]].value
        sections[section]._diam = cell_params[indices[1]].value
        sections[section]._cm = dendrite_cm
        sections[section]._Ra = dendrite_Ra


def _update_synapse_cell_params(net, cell_param_key, param_list):
    cell_params = param_list
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"
    network_synapses = net.cell_types[cell_type]["cell_object"].synapses
    synapse_sections = ["ampa", "nmda", "gabaa", "gabab"]

    param_indices = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]

    # Update Dendrite
    for section, indices in zip(synapse_sections, param_indices):
        network_synapses[section]["e"] = cell_params[indices[0]].value
        network_synapses[section]["tau1"] = cell_params[indices[1]].value
        network_synapses[section]["tau2"] = cell_params[indices[2]].value


def _update_L2_biophysics_cell_params(net, cell_param_key, param_list):
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"
    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    mechs_params = {
        "hh2": {
            "gkbar_hh2": param_list[0].value,
            "gnabar_hh2": param_list[1].value,
            "el_hh2": param_list[2].value,
            "gl_hh2": param_list[3].value,
        },
        "km": {"gbar_km": param_list[4].value},
    }

    sections["soma"].mechs.update(mechs_params)

    # dendrites
    mechs_params["hh2"] = {
        "gkbar_hh2": param_list[5].value,
        "gnabar_hh2": param_list[6].value,
        "el_hh2": param_list[7].value,
        "gl_hh2": param_list[8].value,
    }
    mechs_params["km"] = {"gbar_km": param_list[9].value}

    update_common_dendrite_sections(sections, mechs_params)


def _update_L5_biophysics_cell_params(net, cell_param_key, param_list):
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"
    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    mechs_params = {
        "hh2": {
            "gkbar_hh2": param_list[0].value,
            "gnabar_hh2": param_list[1].value,
            "el_hh2": param_list[2].value,
            "gl_hh2": param_list[3].value,
        },
        "ca": {"gbar_ca": param_list[4].value},
        "cad": {"taur_cad": param_list[5].value},
        "kca": {"gbar_kca": param_list[6].value},
        "km": {"gbar_km": param_list[7].value},
        "cat": {"gbar_cat": param_list[8].value},
        "ar": {"gbar_ar": param_list[9].value},
    }

    sections["soma"].mechs.update(mechs_params)

    # dendrites
    mechs_params["hh2"] = {
        "gkbar_hh2": param_list[10].value,
        "gnabar_hh2": param_list[11].value,
        "el_hh2": param_list[12].value,
        "gl_hh2": param_list[13].value,
    }

    mechs_params["ca"] = {"gbar_ca": param_list[14].value}
    mechs_params["cad"] = {"taur_cad": param_list[15].value}
    mechs_params["kca"] = {"gbar_kca": param_list[16].value}
    mechs_params["km"] = {"gbar_km": param_list[17].value}
    mechs_params["cat"] = {"gbar_cat": param_list[18].value}
    mechs_params["ar"] = {
        "gbar_ar": partial(
            _exp_g_at_dist, zero_val=param_list[19].value, exp_term=3e-3, offset=0.0
        )
    }

    update_common_dendrite_sections(sections, mechs_params)


def update_common_dendrite_sections(sections, mechs_params):
    dendrite_sections = [name for name in sections.keys() if name != "soma"]
    for section in dendrite_sections:
        sections[section].mechs.update(deepcopy(mechs_params))


def _serialize_simulation(log_out, sim_data, simulation_list_widget):
    # Only download if there is at least one simulation
    sim_name = simulation_list_widget.value

    with log_out:
        return serialize_simulation(sim_data, sim_name)


def serialize_simulation(simulations_data, simulation_name):
    """Serializes simulation data to CSV.

    Creates a single CSV file or a ZIP file containing multiple CSVs,
    depending on the number of trials in the simulation.

    """
    simulation_data = simulations_data["simulation_data"]
    csv_trials_output = []
    # CSV file headers
    headers = "times,agg,L2,L5"
    fmt = "%f, %f, %f, %f"

    for dpl_trial in simulation_data[simulation_name]["dpls"]:
        # Combine all data columns at once
        signals_matrix = np.column_stack(
            (
                dpl_trial.times,
                dpl_trial.data["agg"],
                dpl_trial.data["L2"],
                dpl_trial.data["L5"],
            )
        )

        # Using StringIO to collect CSV data
        with io.StringIO() as output:
            np.savetxt(output, signals_matrix, delimiter=",", header=headers, fmt=fmt)
            csv_trials_output.append(output.getvalue())

    if len(csv_trials_output) == 1:
        # Return a single csv file
        return csv_trials_output[0], ".csv"
    else:
        # Create zip file
        return _create_zip(csv_trials_output, simulation_name), ".zip"


def _serialize_config(log_out, sim_data, simulation_list_widget):
    # Only download if there is at least one simulation
    sim_name = simulation_list_widget.value

    with log_out:
        return serialize_config(sim_data, sim_name)


def serialize_config(simulations_data, simulation_name):
    """Serializes Network configuration data to json."""

    # Get network from data dictionary
    net = simulations_data["simulation_data"][simulation_name]["net"]

    # Write to buffer
    with io.StringIO() as output:
        write_network_configuration(net, output)
        return output.getvalue()


def _create_zip(csv_data_list, simulation_name):
    # Zip all files and keep it in memory
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for index, csv_data in enumerate(csv_data_list):
                zf.writestr(f"{simulation_name}_{index + 1}.csv", csv_data)
        zip_buffer.seek(0)
        return zip_buffer.read()


def handle_backend_change(backend_type, backend_config, mpi_cmd, n_jobs):
    """Switch backends between MPI and Joblib."""
    backend_config.clear_output()
    with backend_config:
        if backend_type == "MPI":
            display(VBox(children=[n_jobs, mpi_cmd]))
        elif backend_type == "Joblib":
            display(n_jobs)


def _is_valid_add_tonic_input(drive_widgets):
    for drive in drive_widgets:
        if drive["type"] == "Tonic":
            return False
    return True


def _get_rhythmic_widget_for_opt(
    name,
    tstop_widget,
    layout,
    style,
    location,
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
):
    default_data = {
        "tstart": 0.0,
        "tstart_std": 0.0,
        "tstop": tstop_widget.value,
        "burst_rate": 7.5,
        "burst_std": 0,
        "numspikes": 1,
        "n_drive_cells": 1,
        "cell_specific": False,
        "seedcore": 14,
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    default_data = _update_nested_dict(default_data, data)

    kwargs = dict(layout=layout, style=style)
    tstart = BoundedFloatText(
        value=default_data["tstart"],
        description="Start time (ms)",
        min=0,
        max=1e6,
        **kwargs,
    )
    tstart_std = BoundedFloatText(
        value=default_data["tstart_std"],
        description="Start time dev (ms)",
        min=0,
        max=1e6,
        **kwargs,
    )
    tstop = BoundedFloatText(
        value=default_data["tstop"],
        description="Stop time (ms)",
        max=tstop_widget.value,
        **kwargs,
    )
    burst_rate = BoundedFloatText(
        value=default_data["burst_rate"],
        description="Burst rate (Hz)",
        min=0,
        max=1e6,
        **kwargs,
    )
    burst_std = BoundedFloatText(
        value=default_data["burst_std"],
        description="Burst std dev (Hz)",
        min=0,
        max=1e6,
        **kwargs,
    )
    numspikes = BoundedIntText(
        value=default_data["numspikes"],
        description="No. Spikes:",
        min=0,
        max=int(1e6),
        **kwargs,
    )
    n_drive_cells = IntText(
        value=default_data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=default_data["cell_specific"],
        **kwargs,
    )
    cell_specific = Checkbox(
        value=default_data["cell_specific"], description="Cell-Specific", **kwargs
    )
    seedcore = IntText(value=default_data["seedcore"], description="Seed", **kwargs)

    widgets_list, widgets_dict = _get_drive_weight_widgets(
        layout,
        style,
        location,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
    )

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    opt_drive_box = VBox(
        [
            tstart,
            tstart_std,
            tstop,
            burst_rate,
            burst_std,
            numspikes,
            n_drive_cells,
            cell_specific,
            seedcore,
        ]
        + widgets_list
    )

    opt_drive_widget = dict(
        type="Rhythmic",
        name=name,
        tstart=tstart,
        tstart_std=tstart_std,
        burst_rate=burst_rate,
        burst_std=burst_std,
        numspikes=numspikes,
        seedcore=seedcore,
        location=location,
        tstop=tstop,
        n_drive_cells=n_drive_cells,
        is_cell_specific=cell_specific,
    )
    opt_drive_widget.update(widgets_dict)

    return opt_drive_box, opt_drive_widget


def _get_poisson_widget_for_opt(
    name,
    tstop_widget,
    layout,
    style,
    location,
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
):
    default_data = {
        "tstart": 0.0,
        "tstop": tstop_widget.value,
        "n_drive_cells": 1,
        "cell_specific": True,
        "seedcore": 14,
        "rate_constant": {
            "L2_pyramidal": 40.0,
            "L5_pyramidal": 40.0,
            "L2_basket": 40.0,
            "L5_basket": 40.0,
        },
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    default_data = _update_nested_dict(default_data, data)

    tstart = BoundedFloatText(
        value=default_data["tstart"],
        description="Start time (ms)",
        min=0,
        max=1e6,
        layout=layout,
        style=style,
    )
    tstop = BoundedFloatText(
        value=default_data["tstop"],
        max=tstop_widget.value,
        description="Stop time (ms)",
        layout=layout,
        style=style,
    )
    n_drive_cells = IntText(
        value=default_data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=default_data["cell_specific"],
        layout=layout,
        style=style,
    )
    cell_specific = Checkbox(
        value=default_data["cell_specific"],
        description="Cell-Specific",
        layout=layout,
        style=style,
    )
    seedcore = IntText(
        value=default_data["seedcore"], description="Seed", layout=layout, style=style
    )

    cell_types = ["L5_pyramidal", "L2_pyramidal", "L5_basket", "L2_basket"]
    rate_constant = dict()
    for cell_type in cell_types:
        rate_constant[f"{cell_type}"] = BoundedFloatText(
            value=default_data["rate_constant"][cell_type],
            description=f"{cell_type}:",
            min=0,
            max=1e6,
            step=0.01,
            layout=layout,
            style=style,
        )

    widgets_list, widgets_dict = _get_drive_weight_widgets(
        layout,
        style,
        location,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
    )
    widgets_dict.update({"rate_constant": rate_constant})
    widgets_list.extend(
        [HTML(value="<b>Rate constants</b>")]
        + list(widgets_dict["rate_constant"].values())
    )

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    opt_drive_box = VBox(
        [tstart, tstop, n_drive_cells, cell_specific, seedcore] + widgets_list
    )
    opt_drive_widget = dict(
        type="Poisson",
        name=name,
        tstart=tstart,
        tstop=tstop,
        rate_constant=rate_constant,
        seedcore=seedcore,
        location=location,  # notice this is a widget but a str!
        n_drive_cells=n_drive_cells,
        is_cell_specific=cell_specific,
    )
    opt_drive_widget.update(widgets_dict)

    return opt_drive_box, opt_drive_widget


def _create_opt_widgets_for_var(
    var_name,
    initial_value,
    var_description,
    initial_constraint_range_proportion=None,
    var_layout=None,
    var_style=None,
    checkbox_layout=None,
    checkbox_style=None,
    minmax_layout=None,
    minmax_style=None,
    var_type=float(),
    init_bool=False,
):
    opt_checkbox_widget = Checkbox(
        value=init_bool,
        layout=checkbox_layout,
        style=checkbox_style,
    )
    if isinstance(var_type, float):
        var_widget = BoundedFloatText(
            value=initial_value,
            description=var_description,
            min=0,
            max=1e6,
            step=0.01,
            disabled=True,  # ghosted!
            layout=var_layout,
            style=var_style,
        )
        opt_min_widget = BoundedFloatText(
            value=(initial_value * (1 - initial_constraint_range_proportion)),
            description="Min:",
            min=0,
            max=1e6,
            step=0.01,
            layout=minmax_layout,
            style=minmax_style,
        )
        opt_max_widget = BoundedFloatText(
            value=(initial_value * (1 + initial_constraint_range_proportion)),
            description="Max:",
            min=0,
            max=1e6,
            step=0.01,
            layout=minmax_layout,
            style=minmax_style,
        )
    elif isinstance(var_type, int):
        var_widget = IntText(
            value=initial_value,
            description=var_description,
            layout=var_layout,
            style=var_style,
        )
        opt_min_widget = IntText(
            value=round(initial_value * (1 - initial_constraint_range_proportion)),
            description="Min:",
            layout=minmax_layout,
            style=minmax_style,
        )
        opt_max_widget = IntText(
            value=round(initial_value * (1 + initial_constraint_range_proportion)),
            description="Max:",
            layout=minmax_layout,
            style=minmax_style,
        )
    return {
        f"{var_name}": var_widget,
        f"{var_name}_opt_checkbox": opt_checkbox_widget,
        f"{var_name}_opt_min": opt_min_widget,
        f"{var_name}_opt_max": opt_max_widget,
    }


def _create_hbox_for_opt_var(var_name, widget_dict, layout):
    """Home Box Office widget."""
    return HBox(
        [
            widget_dict[f"{var_name}"],
            widget_dict[f"{var_name}_opt_checkbox"],
            widget_dict[f"{var_name}_opt_min"],
            widget_dict[f"{var_name}_opt_max"],
        ],
        layout=layout,
    )


# AES todo change name
def _get_drive_weight_widgets_for_opt(
    layout,
    style,
    location,
    data=None,
    quadruple_entry_hbox=None,
    **_autogen_opt_widget_kwargs,
):
    default_data = {
        "weights_ampa": {
            "L5_pyramidal": 0.0,
            "L2_pyramidal": 0.0,
            "L5_basket": 0.0,
            "L2_basket": 0.0,
        },
        "weights_nmda": {
            "L5_pyramidal": 0.0,
            "L2_pyramidal": 0.0,
            "L5_basket": 0.0,
            "L2_basket": 0.0,
        },
        "delays": {
            "L5_pyramidal": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 0.1,
            "L2_basket": 0.1,
        },
    }
    if isinstance(data, dict):
        default_data = _update_nested_dict(default_data, data)

    cell_types = ["L5_pyramidal", "L2_pyramidal", "L5_basket", "L2_basket"]
    if location == "distal":
        cell_types.remove("L5_basket")

    # weights_ampa, weights_nmda, delays = dict(), dict(), dict()

    # AES TODO oof, next todo is working out weights. Instead of
    # changing structure of "weights_ampa" dict, probably need to simply
    # create other dicts that have the same per-celltype structure...

    syn_widgets_dict = {
        "weights_ampa": {},
        "weights_nmda": {},
        "delays": {},
    }
    ampa_weights_list, nmda_weights_list, delays_list = [], [], []

    kwargs = dict(layout=layout, style=style)
    weights_ampa, weights_nmda, delays = dict(), dict(), dict()

    for cell_type in cell_types:
        # # AES original method
        # weights_ampa[f"{cell_type}"] = BoundedFloatText(
        #     value=default_data["weights_ampa"][cell_type],
        #     description=f"{cell_type}:",
        #     min=0,
        #     max=1e6,
        #     step=0.01,
        #     **kwargs,
        # )
        # weights_nmda[f"{cell_type}"] = BoundedFloatText(
        #     value=default_data["weights_nmda"][cell_type],
        #     description=f"{cell_type}:",
        #     min=0,
        #     max=1e6,
        #     step=0.01,
        #     **kwargs,
        # )
        # delays[f"{cell_type}"] = BoundedFloatText(
        #     value=default_data["delays"][cell_type],
        #     description=f"{cell_type}:",
        #     min=0,
        #     max=1e6,
        #     step=0.1,
        #     **kwargs,
        # )

        # widgets_dict = {
        #     "weights_ampa": weights_ampa,
        #     "weights_nmda": weights_nmda,
        #     "delays": delays,
        # }

        # AES second attempt
        syn_widgets_dict["weights_ampa"].update(
            _create_opt_widgets_for_var(
                cell_type,
                default_data["weights_ampa"][cell_type],
                f"{cell_type}:",
                **_autogen_opt_widget_kwargs,
            )
        )
        ampa_weights_list.append(
            _create_hbox_for_opt_var(
                cell_type,
                syn_widgets_dict["weights_ampa"],
                quadruple_entry_hbox,
            )
        )
        # weights_nmda.update(
        syn_widgets_dict["weights_nmda"].update(
            _create_opt_widgets_for_var(
                cell_type,
                default_data["weights_nmda"][cell_type],
                f"{cell_type}:",
                **_autogen_opt_widget_kwargs,
            )
        )
        nmda_weights_list.append(
            _create_hbox_for_opt_var(
                cell_type,
                syn_widgets_dict["weights_nmda"],
                quadruple_entry_hbox,
            )
        )

        # delays.update(
        syn_widgets_dict["delays"].update(
            _create_opt_widgets_for_var(
                cell_type,
                default_data["delays"][cell_type],
                f"{cell_type}:",
                **_autogen_opt_widget_kwargs,
            )
        )
        delays_list.append(
            _create_hbox_for_opt_var(
                cell_type,
                syn_widgets_dict["delays"],
                quadruple_entry_hbox,
            )
        )

        # # AES: first attempt at rewriting dicts
        # syn_widgets_dict.update(
        #     _create_opt_widgets_for_var(
        #         f"{cell_type}_weights_ampa",
        #         default_data["weights_ampa"][cell_type],
        #         f"{cell_type}:",
        #         **_autogen_opt_widget_kwargs,
        #     )
        # )
        # ampa_weights_list.append(
        #     _create_hbox_for_opt_var(
        #         f"{cell_type}_weights_ampa",
        #         syn_widgets_dict,
        #         quadruple_entry_hbox,
        #     ))
        # # weights_nmda.update(
        # syn_widgets_dict.update(
        #     _create_opt_widgets_for_var(
        #         f"{cell_type}_weights_nmda",
        #         default_data["weights_nmda"][cell_type],
        #         f"{cell_type}:",
        #         **_autogen_opt_widget_kwargs,
        #     )
        # )
        # nmda_weights_list.append(
        #     _create_hbox_for_opt_var(
        #         f"{cell_type}_weights_nmda",
        #         syn_widgets_dict,
        #         quadruple_entry_hbox,
        #     ))

        # # delays.update(
        # syn_widgets_dict.update(
        #     _create_opt_widgets_for_var(
        #         f"{cell_type}_delays",
        #         default_data["delays"][cell_type],
        #         f"{cell_type}:",
        #         **_autogen_opt_widget_kwargs,
        #     )
        # )
        # delays_list.append(
        #     _create_hbox_for_opt_var(
        #         f"{cell_type}_delays",
        #         syn_widgets_dict,
        #         quadruple_entry_hbox,
        #     ))

        # # AES don't remember what this is from
        # weights_ampa[f"{cell_type}"] = BoundedFloatText(
        #     value=default_data["weights_ampa"][cell_type],
        #     description=f"{cell_type}:",
        #     min=0,
        #     max=1e6,
        #     step=0.01,
        #     **kwargs,
        # )
        # weights_nmda[f"{cell_type}"] = BoundedFloatText(
        #     value=default_data["weights_nmda"][cell_type],
        #     description=f"{cell_type}:",
        #     min=0,
        #     max=1e6,
        #     step=0.01,
        #     **kwargs,
        # )
        # delays[f"{cell_type}"] = BoundedFloatText(
        #     value=default_data["delays"][cell_type],
        #     description=f"{cell_type}:",
        #     min=0,
        #     max=1e6,
        #     step=0.1,
        #     **kwargs,
        # )

    # widgets_dict = {
    #     "weights_ampa": weights_ampa,
    #     "weights_nmda": weights_nmda,
    #     "delays": delays,
    # }
    syn_widgets_list = (
        [HTML(value="<b>AMPA weights</b>")]
        + ampa_weights_list
        + [HTML(value="<b>NMDA weights</b>")]
        + nmda_weights_list
        + [HTML(value="<b>Synaptic delays</b>")]
        + delays_list
    )
    return syn_widgets_list, syn_widgets_dict
    # return widgets_dict_out


def _get_evoked_widget_for_opt(
    name,
    layout,
    style,
    location,
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
):
    # AES TODO: remove top-padding inside, since it's awkward space between drive name and first HTML element
    initial_constraint_range_proportion = 0.2
    default_data = {
        "mu": 0,
        "sigma": 1,
        "numspikes": 1,
        "n_drive_cells": 1,
        "cell_specific": True,
        "seedcore": 14,
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    default_data = _update_nested_dict(default_data, data)

    # Visual config for "main variable" widgets
    var_layout = Layout(width="225px")
    var_style = {"description_width": "100px"}
    # Visual config for checkbox widgets
    checkbox_layout = Layout(width="30px")
    checkbox_style = {"description_width": "0px"}
    # Visual config for min and max constraint widgets
    minmax_layout = Layout(width="100px")
    minmax_style = {"description_width": "30px"}

    quadruple_entry_hbox = Layout(
        display="flex",
        flex_flow="row",
        # align_items='stretch',
        align_items="flex-start",
        # width='90%',
        width="480px",  # AES NO TOUCHING!
        # width='200px',
    )

    # kwargs = dict(layout=layout, style=style)
    # AES TODO write lambda/whatever to multiply and format output of min/max
    html_tab = "&emsp;"

    column_titles = HTML(
        value=f"""
        <div style='margin:0px 0px 0px 190px;'><b>Optimize against?</b>
        {html_tab}{html_tab}{html_tab}Constraints:</div>
        """,
    )

    _autogen_opt_widget_kwargs = dict(
        initial_constraint_range_proportion=initial_constraint_range_proportion,
        var_layout=var_layout,
        var_style=var_style,
        checkbox_layout=checkbox_layout,
        checkbox_style=checkbox_style,
        minmax_layout=minmax_layout,
        minmax_style=minmax_style,
    )

    # AES maybe make dictionary, THEN make opt_drive_box so as to not use var names?
    # opt_drive_widget = dict(
    #     type="Evoked",
    #     name=name,
    #     mu=mu,
    #     mu_opt_min=mu_opt_min,
    #     mu_opt_max=mu_opt_max,
    #     mu_opt_checkbox=mu_opt_checkbox,
    #     sigma=sigma,
    #     numspikes=numspikes,
    #     seedcore=seedcore,
    #     location=location,
    #     sync_within_trial=False,
    #     n_drive_cells=n_drive_cells,
    #     is_cell_specific=cell_specific,
    # )

    # AES TODO observe on these
    opt_drive_widget = dict(
        type="Evoked",
        name=name,
    )
    # mu = BoundedFloatText(
    #     value=default_data["mu"],
    #     description="Mean time:",
    #     min=0,
    #     max=1e6,
    #     step=0.01,
    #     disabled=True,  # ghosted!
    #     layout=var_layout,
    #     style=var_style,
    # )
    opt_drive_widget.update(
        _create_opt_widgets_for_var(
            "mu", default_data["mu"], "Mean time:", **_autogen_opt_widget_kwargs
        )
        | _create_opt_widgets_for_var(
            "sigma",
            default_data["sigma"],
            "Std dev time:",
            **_autogen_opt_widget_kwargs,
        )
        | _create_opt_widgets_for_var(
            "numspikes",
            default_data["numspikes"],
            "No. Spikes:",
            **_autogen_opt_widget_kwargs,
            var_type=int(),
        )
    )

    # sigma = BoundedFloatText(
    #     value=default_data["sigma"],
    #     description="Std dev time:",
    #     min=0,
    #     max=1e6,
    #     step=0.01,
    #     layout=var_layout,
    #     style=var_style,
    # )

    # numspikes = IntText(
    #     value=default_data["numspikes"],
    #     description="No. Spikes:",
    #     layout=var_layout,
    #     style=var_style,
    # )
    n_drive_cells = IntText(
        value=default_data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=default_data["cell_specific"],
        layout=var_layout,
        style=var_style,
    )
    cell_specific = Checkbox(
        value=default_data["cell_specific"],
        description="Cell-Specific",
        layout=var_layout,
        style=var_style,
    )
    seedcore = IntText(
        value=default_data["seedcore"],
        description="Seed: ",
        layout=var_layout,
        style=var_style,
    )
    opt_drive_widget.update(
        dict(
            seedcore=seedcore,
            location=location,
            sync_within_trial=False,
            n_drive_cells=n_drive_cells,
            is_cell_specific=cell_specific,
        )
    )

    syn_widgets_list, syn_widgets_dict = _get_drive_weight_widgets_for_opt(
        var_layout,
        var_style,
        location,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
        quadruple_entry_hbox=quadruple_entry_hbox,
        **_autogen_opt_widget_kwargs,
    )

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    # AEs just insert from the existing dict, duh
    opt_drive_box = VBox(
        [
            column_titles,
            _create_hbox_for_opt_var("mu", opt_drive_widget, quadruple_entry_hbox),
            _create_hbox_for_opt_var("sigma", opt_drive_widget, quadruple_entry_hbox),
            _create_hbox_for_opt_var(
                "numspikes", opt_drive_widget, quadruple_entry_hbox
            ),
            n_drive_cells,
            cell_specific,
            seedcore,
        ]
        + syn_widgets_list
    )

    # AES what to do about this
    opt_drive_widget.update(syn_widgets_dict)
    return opt_drive_box, opt_drive_widget


def _get_tonic_widget_for_opt(name, tstop_widget, layout, style, data=None):
    cell_types = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]
    default_values = {"amplitude": 0, "t0": 0, "tstop": tstop_widget.value}
    t0 = default_values["t0"]
    tstop = default_values["tstop"]
    default_data = {cell_type: default_values for cell_type in cell_types}

    kwargs = dict(layout=layout, style=style)
    if isinstance(data, dict):
        default_data = _update_nested_dict(default_data, data)

    amplitudes = dict()
    for cell_type in cell_types:
        amplitude = default_data[cell_type]["amplitude"]
        amplitudes[cell_type] = BoundedFloatText(
            value=amplitude, description=cell_type, min=0, max=1e6, step=0.01, **kwargs
        )
        # Reset the global t0 and stop with values from the 'data' keyword.
        # It should be same across all the cell-types.
        if amplitude > 0:
            t0 = default_data[cell_type]["t0"]
            tstop = default_data[cell_type]["tstop"]

    start_times = BoundedFloatText(
        value=t0, description="Start time", min=0, max=1e6, step=1.0, **kwargs
    )
    stop_times = BoundedFloatText(
        value=tstop, description="Stop time", min=-1, max=1e6, step=1.0, **kwargs
    )

    widgets_dict = {"amplitude": amplitudes, "t0": start_times, "tstop": stop_times}
    widgets_list = (
        [HTML(value="<b>Times (ms):</b>")]
        + [start_times, stop_times]
        + [HTML(value="<b>Amplitude (nA):</b>")]
        + list(amplitudes.values())
    )

    opt_drive_box = VBox(widgets_list)
    opt_drive_widget = dict(
        type="Tonic",
        name=name,
        amplitude=amplitudes,
        t0=start_times,
        tstop=stop_times,
    )

    opt_drive_widget.update(widgets_dict)

    return opt_drive_box, opt_drive_widget


def _build_opt_objects(
    drive_type,
    name,
    tstop_widget,
    layout,
    style,
    location,
    drive_data,
    weights_ampa,
    weights_nmda,
    delays,
    n_drive_cells,
    cell_specific,
):
    if drive_type in ("Rhythmic", "Bursty"):
        opt_drive_box, opt_drive_widget = _get_rhythmic_widget_for_opt(
            name,
            tstop_widget,
            layout,
            style,
            location,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
        )
    elif drive_type == "Poisson":
        opt_drive_box, opt_drive_widget = _get_poisson_widget_for_opt(
            name,
            tstop_widget,
            layout,
            style,
            location,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
        )
    elif drive_type in ("Evoked", "Gaussian"):
        opt_drive_box, opt_drive_widget = _get_evoked_widget_for_opt(
            name,
            layout,
            style,
            location,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
        )
    elif drive_type == "Tonic":
        opt_drive_box, opt_drive_widget = _get_tonic_widget_for_opt(
            name, tstop_widget, layout, style, data=drive_data
        )
    else:
        raise ValueError(f"Unknown drive type {drive_type}")

    return opt_drive_box, opt_drive_widget


def generate_constraints_and_func(net, opt_drive_widgets):
    # AES TODO params needs to be created dynamically based on which parameters
    # are checked
    # TODO this also means we have to dynamically create the variable names
    #
    constraints = {}
    # First, iterate through set of variable-specific widgets for each drive, assemble
    # param var names, and grab constraint values for those whose checkbox is true. This
    # builds a `constraints` dictionary that is FLAT, where the keys are long variable
    # names (with their context) for which the user has checked the checkbox, and their
    # values are a tuple with their min and max constraints.
    # ------------------------------------------------------------------------------
    for drive_idx, drive in enumerate(opt_drive_widgets):
        if drive["type"] in ("Tonic"):
            # weights_amplitudes = _drive_widget_to_dict(drive, "amplitude")
            # net.add_tonic_bias(
            #     amplitude=weights_amplitudes,
            #     t0=drive["t0"].value,
            #     tstop=drive["tstop"].value,
            # )
            pass
        else:
            # sync_inputs_kwargs = dict(
            #     n_drive_cells=(
            #         "n_cells"
            #         if drive["is_cell_specific"].value
            #         else drive["n_drive_cells"].value
            #     ),
            #     cell_specific=drive["is_cell_specific"].value,
            # )

            # weights_ampa = _drive_widget_to_dict(drive, "weights_ampa")
            # weights_nmda = _drive_widget_to_dict(drive, "weights_nmda")
            # synaptic_delays = _drive_widget_to_dict(drive, "delays")
            # print(f"drive type is {drive['type']}, location={drive['location']}")

            # Synaptic variables are a special case, since they are dicts instead of
            # single values
            for syn_type in ("weights_ampa", "weights_nmda", "delays"):
                for key in drive[syn_type].keys():
                    # For every variable with a checkbox, but only if the checkbox
                    # is true/checked
                    if ("_opt_checkbox" in key) and (drive[syn_type][key].value):
                        # Extract the var name, which in the complicated synaptic
                        # case is ONLY the celltype
                        var_name = key.split("_opt_checkbox")[0]
                        # Create a new, unique var name for this drive's instance of
                        # that variable, which will become our key in our
                        # `constraints` dict. Since we are dealing with synaptic
                        # variables, we ALSO need to add the type of weight/delay:
                        unique_param_name = str(
                            drive["type"]
                            + "_"
                            + drive["name"]
                            + "_"
                            + syn_type
                            + "_"
                            + var_name
                        )
                        # Use the unique name as the key, and add the bounds
                        constraints.update(
                            {
                                unique_param_name: tuple(
                                    [
                                        drive[syn_type][var_name + "_opt_min"].value,
                                        drive[syn_type][var_name + "_opt_max"].value,
                                    ]
                                )
                            }
                        )

            if drive["type"] == "Poisson":
                rate_constant = _drive_widget_to_dict(drive, "rate_constant")

            elif drive["type"] in ("Evoked", "Gaussian"):
                for key in drive.keys():
                    # For every variable with a checkbox, but only if the checkbox
                    # is true/checked
                    if ("_opt_checkbox" in key) and (drive[key].value):
                        # Extract the var name
                        var_name = key.split("_opt_checkbox")[0]
                        # Create a new, unique var name for this drive's instance of
                        # that variable, which will become our key in our
                        # `constraints` dict
                        unique_param_name = str(
                            drive["type"] + "_" + drive["name"] + "_" + var_name
                        )
                        # Use the unique name as the key, and add the bounds
                        constraints.update(
                            {
                                unique_param_name: tuple(
                                    [
                                        drive[var_name + "_opt_min"].value,
                                        drive[var_name + "_opt_max"].value,
                                    ]
                                )
                            }
                        )
            elif drive["type"] in ("Rhythmic", "Bursty"):
                pass

    # Second, create a new `set_params` function that iterates through the drive
    # widgets AGAIN, but which deploys our newly-created `constraints` dict:
    # ------------------------------------------------------------------------------
    def set_params(net, params):
        for drive_idx, drive in enumerate(opt_drive_widgets):

            def name_check(var, syn_type=None):
                unique_param_name = str(
                    drive["type"]
                    + "_"
                    + drive["name"]
                    + "_"
                    + (syn_type + "_" if syn_type else "")
                    + var
                )
                if unique_param_name in params.keys():
                    return unique_param_name
                else:
                    return None

            def use_params_if_exists(var_name):
                return (
                    params[name_check(var_name)]
                    if name_check(var_name)
                    else drive[var_name].value
                )

            if drive["type"] in ("Tonic"):
                # weights_amplitudes = _drive_widget_to_dict(drive, "amplitude")
                # net.add_tonic_bias(
                #     amplitude=weights_amplitudes,
                #     t0=drive["t0"].value,
                #     tstop=drive["tstop"].value,
                # )
                pass
            else:
                sync_inputs_kwargs = dict(
                    n_drive_cells=(
                        "n_cells"
                        if drive["is_cell_specific"].value
                        else drive["n_drive_cells"].value
                    ),
                    cell_specific=drive["is_cell_specific"].value,
                )

                deployed_syn_dicts = {
                    "weights_ampa": {},
                    "weights_nmda": {},
                    "delays": {},
                }
                cell_types = [
                    "L5_pyramidal",
                    "L2_pyramidal",
                    "L5_basket",
                    "L2_basket",
                ]
                if drive["location"] == "distal":
                    cell_types.remove("L5_basket")

                for syn_type in deployed_syn_dicts:
                    for ct in cell_types:
                        deployed_syn_dicts[syn_type].update(
                            {
                                ct: (
                                    params[name_check(ct, syn_type)]
                                    if name_check(ct, syn_type)
                                    else drive[syn_type][ct].value
                                )
                            }
                        )

                print(f"drive type is {drive['type']}, location={drive['location']}")
                if drive["type"] == "Poisson":
                    # AES TODO rate constants need to be treated like other per-celltype synaptic thingies
                    rate_constant = _drive_widget_to_dict(drive, "rate_constant")
                    net.add_poisson_drive(
                        name=drive["name"],
                        tstart=drive["tstart"].value,
                        tstop=drive["tstop"].value,
                        rate_constant=rate_constant,
                        location=drive["location"],
                        weights_ampa=deployed_syn_dicts["weights_ampa"],
                        weights_nmda=deployed_syn_dicts["weights_nmda"],
                        synaptic_delays=deployed_syn_dicts["delays"],
                        space_constant=100.0,
                        event_seed=drive["seedcore"].value,
                        **sync_inputs_kwargs,
                    )
                elif drive["type"] in ("Evoked", "Gaussian"):
                    net.add_evoked_drive(
                        name=drive["name"],
                        mu=use_params_if_exists("mu"),
                        sigma=drive["sigma"].value,
                        numspikes=drive["numspikes"].value,
                        location=drive["location"],
                        weights_ampa=deployed_syn_dicts["weights_ampa"],
                        weights_nmda=deployed_syn_dicts["weights_nmda"],
                        synaptic_delays=deployed_syn_dicts["delays"],
                        space_constant=3.0,
                        event_seed=drive["seedcore"].value,
                        **sync_inputs_kwargs,
                    )

                elif drive["type"] in ("Rhythmic", "Bursty"):
                    net.add_bursty_drive(
                        name=drive["name"],
                        tstart=drive["tstart"].value,
                        tstart_std=drive["tstart_std"].value,
                        tstop=drive["tstop"].value,
                        location=drive["location"],
                        burst_rate=drive["burst_rate"].value,
                        burst_std=drive["burst_std"].value,
                        numspikes=drive["numspikes"].value,
                        weights_ampa=deployed_syn_dicts["weights_ampa"],
                        weights_nmda=deployed_syn_dicts["weights_nmda"],
                        synaptic_delays=deployed_syn_dicts["delays"],
                        event_seed=drive["seedcore"].value,
                        **sync_inputs_kwargs,
                    )

    return set_params, constraints


def run_opt_button_clicked(
    widget_simulation_name,
    log_out,
    opt_drive_widgets,
    all_data,
    dt,
    tstop,
    fig_default_params,
    widget_default_smoothing,
    widget_default_scaling,
    widget_min_frequency,
    widget_max_frequency,
    ntrials,
    backend_selection,
    mpi_cmd,
    n_jobs,
    params,
    simulation_status_bar,
    simulation_status_contents,
    connectivity_textfields,
    viz_manager,
    simulations_list_widget,
    cell_parameters_widgets,
    global_gain_textfields,
    opt_solver,
    opt_obj_fun,
    opt_max_iter,
    opt_tstop,
    opt_target_data_name,
):
    """Run the simulation and plot outputs.

    This was initially built from copying `run_button_clicked`.
    """
    with log_out:
        # Sim data setup (and related input validation)
        # ------------------------------------------------------------------------------
        simulation_data = all_data["simulation_data"]

        # clear empty trash simulations
        # AES: a "trash" simulation appears to be created (named "default") even if all
        # a user does is load an external dipole data file. However, I do not fully
        # understand how VizManager et al. manages the simulation data (I find it very
        # confusing) so I am NOT touching it.
        for _name in tuple(simulation_data.keys()):
            if len(simulation_data[_name]["dpls"]) == 0:
                del simulation_data[_name]

        # AES TODO UGH need to handle existing sim
        _sim_name = widget_simulation_name.value

        # Target data extraction (and related input validation)
        # ------------------------------------------------------------------------------
        if not opt_target_data_name:
            # In this case, they probably have not run any simulations or loaded any data.
            logger.error(
                textwrap.dedent("""
                You have not selected a dataset to use as the target of
                optimization. Please load and select a dataset of dipole data to
                optimize towards.
                """).replace("\n", " ")
            )
            simulation_status_bar.value = simulation_status_contents["failed"]
            return
        elif (opt_target_data_name == "default") and (
            not simulation_data["default"]["dpls"]
        ):
            # In this case, they have selected "default", which is the default name of
            # the first simulation, BUT they have not actually run any simulations
            # yet. They likely either want to compare against a simulation result, or
            # (more likely) forgot to load their experimental target data first.
            #
            # ATTN: How we want to handle this, and what we want to communicate, needs
            # some discussion and thinking.
            logger.error(
                textwrap.dedent("""
                You have selected the 'default' dataset to use as the target of
                optimization, but there is no dipole data associated with that dataset.
                Please either load and select a dataset of dipole data to optimize
                towards, or run a simulation first if you want to optimize against that
                simulation.
                """).replace("\n", " ")
            )
            simulation_status_bar.value = simulation_status_contents["failed"]
            return
        else:
            # Extract the actual target data
            # Like everywhere else in the GUI, we only support usage of single-trial
            # dipole data.
            target_dipole = simulation_data[opt_target_data_name]["dpls"][0]

        # Input validation
        # ------------------------------------------------------------------------------
        # Note that this function initializes our `Network` object at
        # `simulation_data[_sim_name]['net']`, which we will use later. If a user has
        # previously run a simulation for this `_sim_name`, then this will *overwrite*
        # the `Network` object at that location. The only difference, however, is that
        # drives will not be added at this step, due to `add_drive=False`. We need this
        # because Optimization needs to have complete control over how drives are added,
        # since the drives need to be added in the Optimizer's `set_params` function.
        _init_network_from_widgets(
            params,
            dt,
            tstop,
            simulation_data[_sim_name],
            opt_drive_widgets,
            connectivity_textfields,
            cell_parameters_widgets,
            global_gain_textfields,
            add_drive=False,
        )

        # Dynamically create both the constraints and the param-applying-function
        # ------------------------------------------------------------------------------
        #
        # ------------------------------------------------------------------------------
        # THE AES DEBUGGGGG ZONE
        # AES TODO not working for some reason, investigate
        # Set the middle drive's checkbox off, just to keep things interesting
        # opt_drive_widgets[0]["mu_opt_checkbox"].value = False
        opt_drive_widgets[0]["weights_ampa"]["L2_pyramidal_opt_checkbox"].value = True
        # opt_max_iter = 15
        opt_max_iter = 3
        n_jobs.value = 11

        # # AES for debugging readin
        # # from urllib.request import urlretrieve
        # # data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
        # #             'data/MEG_detection_data/yes_trial_S1_ERP_all_avg.txt')
        # # urlretrieve(data_url, 'yes_trial_S1_ERP_all_avg.txt')
        # from hnn_core import read_dipole
        # target_dipole = read_dipole('yes_trial_S1_ERP_all_avg.txt')
        # # target_dipole = read_dipole('S1_SupraT.txt')
        # # # UGHHHHH. Apparently some of our own dipole outputs can't be used?
        # # target_dipole = read_dipole('dpl2.txt')

        set_params_func, constraints = generate_constraints_and_func(
            simulation_data[_sim_name]["net"],
            opt_drive_widgets,
        )
        if not constraints:
            logger.error(
                textwrap.dedent("""
                You have not selected any parameters to constrain in the optimization.
                Please select some parameters using the checkboxes, and try again.
                """).replace("\n", " ")
            )
            simulation_status_bar.value = simulation_status_contents["failed"]
            return

        # Instantiate our Optimizer object
        # ------------------------------------------------------------------------------
        # This uses our recreated `Network` object (which has NO current drives), our
        # new dynamically-created constraints, and our similarly-created params function
        # to build our Optimizer. All other arguments are simply pulled from their GUI
        # values directly.
        optim = Optimizer(
            initial_net=simulation_data[_sim_name]["net"],
            tstop=opt_tstop,
            constraints=constraints,
            set_params=set_params_func,
            solver=opt_solver,
            obj_fun=opt_obj_fun,
            max_iter=opt_max_iter,
        )

        # Setup simulation backends
        # ------------------------------------------------------------------------------
        if backend_selection.value == "MPI":
            # 'use_hwthreading_if_found' and 'sensible_default_cores' have
            # already been set elsewhere, and do not need to be re-set here.
            # Hardware-threading and oversubscription will always be disabled
            # to prevent edge cases in the GUI.
            backend = MPIBackend(
                n_procs=n_jobs.value,
                mpi_cmd=mpi_cmd.value,
                override_hwthreading_option=False,
                override_oversubscribe_option=False,
            )
        else:
            backend = JoblibBackend(n_jobs=n_jobs.value)
            print(f"Using Joblib with {n_jobs.value} core(s).")

        with backend:
            simulation_status_bar.value = simulation_status_contents["opt_running"]
            logger.info("Optimization started.")
            logger.info(f"Solver: {opt_solver}")
            logger.info(f"Objective function: {opt_obj_fun}")
            logger.info(f"Max iterations: {opt_max_iter}")
            logger.info(f"Simulation duration: {opt_tstop} ms")

            # Execute optimization
            # --------------------------------------------------------------------------
            # AES TODO maximize PSD, which requires even more parameters, yay
            try:
                if opt_obj_fun == "dipole_rmse":
                    optim.fit(target=target_dipole, n_trials=ntrials.value)
            except Exception as e:
                logger.error(
                    f"Optimization fitting failed due to exception: '{e}'",
                    exc_info=True,
                )
                simulation_status_bar.value = simulation_status_contents["failed"]
                raise

            logger.info("Optimization finished!")

            # AES DEBUG mode
            # # Check if optimization showed ANY difference in the objective function. If
            # # it did not, then we made no progress, and there's no point in
            # # re-simulating or displaying the output.
            # if np.all(optim.obj_ == optim.obj_[0]):
            #     logger.error(
            #         textwrap.dedent("""
            #         The objective function did not change over the course of the
            #         optimization. You probably need to increase the number of max
            #         iterations in order to start converging.
            #         """).replace("\n", " ")
            #     )
            #     simulation_status_bar.value = simulation_status_contents["failed"]
            #     return

            # --------------------------------------------------------------------------
            # Now, let's resimulate the final version of the optimized network for usage
            # and display it in the GUI
            #
            # First, let's make our new simulation name.
            if (_sim_name + "_optimized") in simulation_data.keys():
                # Let's handle our output simulation name in the case that there are
                # pre-existing datasets with the names we want to use, such as if they
                # are executing a second round of Optimization, or their simulation name
                # just happens to end in "_optimized" by coincidence:
                if (_sim_name + "_optimized" + "_1") in simulation_data.keys():
                    predecessor_sim_suffix_number = []
                    for key in simulation_data.keys():
                        match = re.search(r"_(\d+)$", key)
                        if match:
                            predecessor_sim_suffix_number.append(int(match.group(1)))
                    new_name = (
                        _sim_name
                        + "_optimized"
                        + f"_{max(predecessor_sim_suffix_number) + 1}"
                    )
                else:
                    new_name = _sim_name + "_optimized" + "_1"
            else:
                new_name = _sim_name + "_optimized"

            # Now let's use the final version of the optimized Network, and use it to
            # simulate
            simulation_data[new_name]["net"] = optim.net_
            simulation_data[new_name]["dpls"] = simulate_dipole(
                simulation_data[new_name]["net"],
                tstop=tstop.value,
                dt=dt.value,
                n_trials=ntrials.value,
            )
            simulation_status_bar.value = simulation_status_contents["finished"]

            # AES TODO Maybe force a file-save of the final network params too
            # automatically? This is a good spot to do it

            # Finally, update the list of simulations to include our new one:
            sim_names = [
                sim_name
                for sim_name in simulation_data
                if simulation_data[sim_name]["net"] is not None
            ]
            simulations_list_widget.options = sim_names
            simulations_list_widget.value = sim_names[0]

        # ------------------------------------------------------------------------------
        # The remainder of this function is just repeating some post-run visualization
        # steps, which are identical to those in `run_button_clicked`
        viz_manager.reset_fig_config_tabs()

        # update default visualization params in gui based on widget
        fig_default_params["default_smoothing"] = widget_default_smoothing.value
        fig_default_params["default_scaling"] = widget_default_scaling.value
        fig_default_params["default_min_frequency"] = widget_min_frequency.value
        fig_default_params["default_max_frequency"] = widget_max_frequency.value

        # change default visualization params in viz_manager to mirror gui
        for widget, value in fig_default_params.items():
            viz_manager.fig_default_params[widget] = value

        viz_manager.add_figure()
        fig_name = _idx2figname(viz_manager.data["fig_idx"]["idx"] - 1)
        ax_plots = [("ax0", "input histogram"), ("ax1", "current dipole")]
        for ax_name, plot_type in ax_plots:
            # AES TODO: for dipole_rmse, auto-plot the target data too
            viz_manager._simulate_edit_figure(
                fig_name, ax_name, (_sim_name + "_optimized"), plot_type, {}, "plot"
            )


def launch():
    """Launch voila with hnn_widget.ipynb.

    You can pass voila commandline parameters as usual.
    """
    from voila.app import main

    notebook_path = Path(__file__).parent / "hnn_widget.ipynb"
    main([str(notebook_path.resolve()), *sys.argv[1:]])
