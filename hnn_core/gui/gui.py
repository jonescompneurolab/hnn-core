"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>
import base64
import codecs
import io
import json
import logging
import mimetypes
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from IPython.display import IFrame, Javascript, display
from ipywidgets import (
    HTML,
    Accordion,
    AppLayout,
    BoundedFloatText,
    BoundedIntText,
    Box,
    Button,
    Checkbox,
    Dropdown,
    FileUpload,
    FloatText,
    HBox,
    IntText,
    Layout,
    Output,
    RadioButtons,
    Tab,
    Text,
    VBox,
)
from ipywidgets.embed import embed_minimal_html

import hnn_core
from hnn_core import JoblibBackend, MPIBackend, simulate_dipole
from hnn_core.cells_default import _exp_g_at_dist
from hnn_core.dipole import _read_dipole_txt
from hnn_core.gui._logging import logger
from hnn_core.gui._viz_manager import _idx2figname, _VizManager
from hnn_core.hnn_io import dict_to_network, write_network_configuration
from hnn_core.network import pick_connection
from hnn_core.parallel_backends import (
    _determine_cores_hwthreading,
    _has_mpi4py,
    _has_psutil,
)
from hnn_core.params_default import get_L2Pyr_params_default, get_L5Pyr_params_default

from ..externals.mne import _validate_type

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
        except Exception:
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
    total_height : int
        The height of the GUI in pixels. For an explanation of the various layout
        windows and how they relate, see the comments in ``HNNGUI.__init__``.
    total_width : int
        The width of the GUI in pixels
    header_height : int
        The height of the header in pixels
    status_height : int
        The height of status bar in pixels
    param_window_width_prop : float
        The proportion of the width reserved for the "parameters-window" container;
        the proportion reserved for the visualization-window is then defined as
        (1 - param_window_width_prop)
    log_window_height_prop : float
        The proportion of the height reserved for the "log-window" container. The
        height of the parameters-window grows to fill the remaining space and is thus
        not specified directly
    dpi : int
        The pixel density specified in dots per inch
    network configuration : str
        The relative path to the hierarchical json file defining the network and
        drives to be used, typically "jones2009_base.json"

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
        total_height=800,
        total_width=1300,
        header_height=42,
        status_height=30,
        param_window_width_prop=0.45,
        log_window_height_prop=0.22,
        dpi=96,
        network_configuration=default_network_configuration,
    ):
        _validate_type(total_height, types="int", item_name="total_height")
        _validate_type(total_width, types="int", item_name="total_width")
        _validate_type(header_height, types="int", item_name="header_height")
        _validate_type(status_height, types="int", item_name="status_height")
        _validate_type(
            param_window_width_prop,
            types="numeric",
            item_name="param_window_width_prop",
        )
        _validate_type(
            log_window_height_prop,
            types="numeric",
            item_name="log_window_height_prop",
        )
        _validate_type(dpi, types="int", item_name="dpi")

        # ----------------------------------------------------------------------
        # Structural overview of the GUI
        # ----------------------------------------------------------------------
        # The HNN GUI is composed using ipywidget's AppLayout, which uses the
        # following structure:
        #   | -------------- header --------------- |
        #   | left-sidebar | center | right-sidebar |
        #   | -------------- footer --------------- |
        #
        # Throughout the following code, we add HTML classes (i.e., "tags") to the key
        # "containers" that hold all of our widgets, and to several of the widgets
        # themselves. These HTML classes are sometimes used to apply custom styling to
        # certain elements.
        #
        # Even when not used for styling, these HTML classes are extremely
        # valuable for debugging at the browser level when you need to inspect page
        # elements directly (via inspect) or programmatically (via the console).
        #
        #     > Note: the value of including custom HTML tags comes from the fact that
        #       ipywidgets and AppLayout necessarily add numerous layers of nested
        #       <div> elements (i.e.,"blocks") to the HTML tree when instantiating
        #       widgets, which would make it extremely difficult to ascertain which
        #       <div> elements we need to target with CSS without the use of HTML
        #       tags to identify the relevant containers
        #
        # Using our custom class names in place of the AppLayout parameter names, the
        # structure of the GUI can similarly be written as follows:
        #   | ----------------- title-bar ----------------- |
        #   |   parameters-window  | | visualization-window |
        #   | ----------------- status-bar ---------------- |
        #
        #     > Note that we do not (currently) utilize the "center" AppLayout
        #       container, which is set to 0px in our AppLayout instantiation below
        #
        # The diagrams below outline the structure of these automatically-generated
        # "outer" parent containers, with our included HTML tags
        #
        # The AppLayout header (title-bar)
        #   ======================= title-bar =======================
        #   ||                                                     ||
        #   ||   ============== title-bar-contents =============   ||
        #   ||   ||                                           ||   ||
        #   ||   ||             ~ contents here ~             ||   ||
        #   ||   ||                                           ||   ||
        #   ||   ===============================================   ||
        #   ||                                                     ||
        #   =========================================================
        #
        # The AppLayout left-sidebar (parameters-window)
        #   =================== parameters-window ===================
        #   ||                                                     ||
        #   ||   ========== param-tabs-outer-container =========   ||
        #   ||   ||                                           ||   ||
        #   ||   ||  ======== param-widget-container =======  ||   ||
        #   ||   ||  ||                                   ||  ||   ||
        #   ||   ||  ||         ~ contents here ~         ||  ||   ||
        #   ||   ||  ||                                   ||  ||   ||
        #   ||   ||  =======================================  ||   ||
        #   ||   ||                                           ||   ||
        #   ||   ===============================================   ||
        #   ||                                                     ||
        #   ||   ================== log-window =================   ||
        #   ||   ||                                           ||   ||
        #   ||   ||             ~ contents here ~             ||   ||
        #   ||   ||                                           ||   ||
        #   ||   ===============================================   ||
        #   ||                                                     ||
        #   =========================================================
        #
        # The AppLayout right-sidebar (visualization-window)
        #   ================== visualization-window =================
        #   ||                                                     ||
        #   ||   ====== [[ nested, auto-generated tags ]] ======   ||
        #   ||   ||                                           ||   ||
        #   ||   ||  ============== fig-tabs ===============  ||   ||
        #   ||   ||  ||                                   ||  ||   ||
        #   ||   ||  ||         ~ contents here ~         ||  ||   ||
        #   ||   ||  ||                                   ||  ||   ||
        #   ||   ||  =======================================  ||   ||
        #   ||   ||                                           ||   ||
        #   ||   ===============================================   ||
        #   ||                                                     ||
        #   =========================================================
        #     > Note: the "fig-tabs" HTML class is applied when the Tab() container
        #       is initialized in _viz_manager.py, and not in gui.py
        #
        # The AppLayout footer (status-bar)
        #   ====================== status-bar =======================
        #   ||                                                     ||
        #   ||   =============== sim-status-box ================   ||
        #   ||   ||                                           ||   ||
        #   ||   ||             ~ contents here ~             ||   ||
        #   ||   ||                                           ||   ||
        #   ||   ===============================================   ||
        #   ||                                                     ||
        #   =========================================================
        #
        # Keep in mind that there are many more HTML tags used that listed in the
        # diagrams above; these are merely the tags that are used to specify the
        # primary "outer" containers that house the actual GUI contents

        # ----------------------------------------------------------------------
        # Set the layout properties for various GUI components
        # ----------------------------------------------------------------------

        # Set up container height / width parameters
        # ----------------------------------------------------------------------
        # Containers properties are computed relative to total height/width,
        # allowing us to scale the GUI size without needed to figure out what the
        # exact pixel values need to be for each element
        self.total_height = total_height
        self.total_width = total_width

        # We'll compute pixels for the "fixed" outer containers (per AppLayout), but
        # we'll be able to use percentages for most of the "inner" containers
        # Note that we must use int() as we cannot have fractional pixel values
        parameters_window_width = int(total_width * param_window_width_prop)
        figures_window_width = int(total_width - parameters_window_width)
        main_content_height = total_height - status_height

        # specify the gap between the footer ("status-bar") and the containers
        # above it
        footer_gap = 10

        # default (shared) height for buttons
        button_height = 30

        self.layout = {
            # ==================================================
            # Elements that are not containers
            # ==================================================
            # dpi is technically used in only one place (the _add_figure function in
            # _viz_manager.py), but it is defined here so that it can be specified
            # when calling ``HNNGUI.__init__`` and be passed to the viz_layout
            # argument via "self.viz_manager = _VizManager(...)" below
            # [REF]: [DSD] this parameter should ideally be separated from self.layout,
            # which should only define the layout properties of containers.
            "dpi": dpi,
            #
            # [REF]: [DSD] imho these elements below, which describe buttons, should
            # also be separated from self.layout. Similar to dpi, "theme_color" and
            # "btn" are passed to viz_layout, albeit they are only used in one
            # place in _viz_manager, when self.make_fig_button is called. This button
            # already has make-fig-btn HTML class, so adding the styling directly via
            # CSS to the different buttons may be the better approach
            "theme_color": "#802989",
            "btn": Layout(
                height=f"{button_height}px",
                width="auto",
            ),
            "run_btn": Layout(
                height=f"{button_height}px",
                width="130px",
            ),
            "save_btn": Layout(
                height=f"{button_height}px",
                width="264px",
            ),
            "btn_full_w": Layout(
                height=f"{button_height}px",
                width="100%",
            ),
            "del_fig_btn": Layout(
                height=f"{button_height}px",
                width="auto",
            ),
            #
            # ==================================================
            # Styling for the header and footer containers
            # ==================================================
            # style for the "header" section of AppLayout that display the GUI
            # title and contains the light-dark toggle button
            #   - associated html class: "title-bar"
            "header_height": f"{header_height}px",
            #
            # style for the "footer" section of AppLayout that shows the simulation
            # status
            #   - associated html class: "status-bar"
            "simulation_status_height": f"{status_height}px",
            #
            # ==================================================
            # Styling for "parameters-window" and its children
            # ==================================================
            # container for parameter specification and simulation output
            # that occupies the "left_sidebar" section in AppLayout
            #   - associated html class: "parameters-window"
            "parameters_window": Layout(
                width=f"{parameters_window_width}px",
                height=f"{main_content_height}px",
            ),
            #
            # this "intermediate" container holds the parameter tabs container
            # as well as the log container
            #   - child of "parameters-window"
            #   - associated html class: "param-tabs-outer-container"
            "param_tabs_outer_container": Layout(
                width="100%",
                height="100%",
            ),
            #
            # this container holds the parameters-window tab bar *and* the associated
            # contents for each tab, which are separate <div> trees under
            # param-tabs-outer-container
            #   - child of "parameters-window" > "param_tabs_outer_container"
            #   - associated html class: "param-window-tabs-widget"
            "param_window_tabs_widget": Layout(
                width="98%",
                height="98%",
                margin="0px 0px 0px 0px",
            ),
            #
            # this container is specific to the *contents* of the parameters tabs
            # widget, and does not include the tab bar. It sets the boundary
            # *exclusively* for the contents of the Simulation tab
            #   - child of "parameters-window" > "param_tabs_outer_container" >
            #     "param-window-tabs-widget" > "widget-tab-contents"
            #   - associated html class: "simulation-tab-contents"
            # note that "widget-tab-contents" is an auto-generated container that
            # we do not specifically tag, but it is often used in conjunction with
            # the parent or child container for targeted CSS styling
            "simulation_tab_contents": Layout(
                width="100%",
                height="100%",
            ),
            #
            # styles the text boxes within the collapsible widgets that contain the
            # instantiated drives in the External Drives tab
            #   - child of "parameters-window" > ... > "drive-tab-contents" >
            #     "widget-output" > ... > "widget-vbox"
            "drive_textbox": Layout(
                width="270px",
                height="auto",
            ),
            #
            # container for the log window
            #   - child of "parameters-window"
            #   - associated html class: "log-window"
            # note: we use a margin on log-window to set the footer-gap here, as the
            # gap is inserted between the inner container "log-window" and its parent
            # container "parameters-window."" Adding the gap as a margin directly to
            # parameters-window would also require recalculating the height of the
            # container to accommodate the extra margin, since its height is fixed
            "log_window": Layout(
                border="1px solid lightgray",
                height=f"{int(log_window_height_prop * 100)}%",
                width="98%",
                margin=f"0px 0px {footer_gap}px 0px",
                overflow="auto",
            ),
            #
            # ==================================================
            # Styling for "visualization-window" and its children
            # ==================================================
            # container for simulation output and visualizations that occupies
            # the "right_sidebar" section in AppLayout
            #   - associated html class: "visualization-window"
            #
            # note: we set the footer-gap here by recalculating the height of
            # visualization-window. Adding footer_gap as a margin, as done above for
            # log-window above, would not shift the content up, as the container height
            # is still specified in pixels. Rather, the margin on visualization-window
            # would simply overflow into the footer
            "visualization_window": Layout(
                width=f"{figures_window_width}px",
                height=f"{main_content_height - footer_gap}px",
                border="1px solid lightgrey",
            ),
            #
            # directly set figsize for the matplotlib figure in the Output()
            # blocks of visualization-window
            #   - child of visualization-window > visualization-output > ... >
            #     ( <img src=...> | <div class="jupyter-matplotlib-figure"...>)
            #   - associated html class: NA
            #
            # note: figure sizes are set in _add_figure in _viz_manager, where
            # percents are converted to pixels. This CSS block sets the dimensions
            # for both static figure outputs (<img src="data:img/png;base64,...>")
            # AND for dynamic figure outputs (<div class="jupyter-matplotlib-figure">)
            "visualization_output_figsize": Layout(
                width="100%",
                height="95%",
            ),
        }

        # Set up for the simulation status bar
        # ----------------------------------------------------------------------
        # we directly set up the html for the status bar below
        # this dict is referenced in _init_ui_components and run_button_clicked
        #   - child of status-bar
        #   - associated html class: sim-status-box
        self._simulation_status_contents = {
            "not_running": """
                <div
                class='sim-status-box'
                style='
                    background:gray;
                    padding-left:10px;
                    color:white;
                '>
                    Not running
                </div>
            """,
            "running": """
                <div
                class='sim-status-box status-running'
                style='
                    background:var(--statusbar-running);
                    padding-left:10px;
                    color:white;
                '>
                    Running...
                </div>
            """,
            "finished": """
                <div
                class='sim-status-box'
                style='
                    background:var(--gentle-green);
                    padding-left:10px;
                    color:white;
                '>
                    Simulation finished
                </div>
            """,
            "failed": """
                <div
                class='sim-status-box'
                style='
                    background:var(--gentle-red);
                    padding-left:10px;
                    color:white;
                '>
                    Simulation failed
                </div>
            """,
        }

        # ----------------------------------------------------------------------
        # Set up the GUI widgets and their contents
        # ----------------------------------------------------------------------
        # load default parameters
        self.params = self.load_parameters(network_configuration)

        # Number of available cores
        [self.n_cores, _] = _determine_cores_hwthreading(
            use_hwthreading_if_found=False,
            sensible_default_cores=True,
        )

        # In-memory storage of all simulation and visualization related data
        self.simulation_data = defaultdict(lambda: dict(net=None, dpls=list()))

        # ==================================================
        # Simulation tab
        # ==================================================

        # input fields for simulation parameters
        # --------------------------------------------------
        self.widget_simulation_name = Text(
            value="default",
            placeholder="ID of your simulation",
            description="Name:",
            disabled=False,
        )
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
        self.widget_backend_selection = Dropdown(
            options=[("Joblib", "Joblib"), ("MPI", "MPI")],
            value=self._check_backend(),
            description="Backend:",
        )
        self.widget_n_jobs = BoundedIntText(
            value=1,
            min=1,
            max=self.n_cores,
            description="Cores:",
            disabled=False,
        )
        self.widget_mpi_cmd = Text(
            value="mpiexec",
            placeholder="Fill if applies",
            description="MPI cmd:",
            disabled=False,
        )

        # input fields for default visualization parameters
        # --------------------------------------------------
        default_param_properties = {
            # container_width includes the space allocated for both the text and
            # the input field
            "container_width": Layout(width="300px"),
            "text_width": {"description_width": "200px"},
        }
        self.widget_default_smoothing = BoundedFloatText(
            value=30.0,
            description="Dipole Smoothing:",
            min=0.0,
            max=100.0,
            step=1.0,
            disabled=False,
            layout=default_param_properties["container_width"],
            style=default_param_properties["text_width"],
        )
        self.widget_default_scaling = FloatText(
            value=3000.0,
            description="Dipole Scaling:",
            step=100.0,
            disabled=False,
            layout=default_param_properties["container_width"],
            style=default_param_properties["text_width"],
        )
        self.widget_min_frequency = BoundedFloatText(
            value=10,
            min=0.1,
            max=1000,
            description="Min Spectral Frequency (Hz):",
            disabled=False,
            layout=default_param_properties["container_width"],
            style=default_param_properties["text_width"],
        )
        self.widget_max_frequency = BoundedFloatText(
            value=100,
            min=0.1,
            max=1000,
            description="Max Spectral Frequency (Hz):",
            disabled=False,
            layout=default_param_properties["container_width"],
            style=default_param_properties["text_width"],
        )
        self.fig_default_params = {
            "default_smoothing": self.widget_default_smoothing.value,
            "default_scaling": self.widget_default_scaling.value,
            "default_min_frequency": self.widget_min_frequency.value,
            "default_max_frequency": self.widget_max_frequency.value,
        }

        # simulation tab buttons
        # --------------------------------------------------
        self.load_data_button = FileUpload(
            accept=".txt,.csv",
            multiple=False,
            style={"button_color": self.layout["theme_color"]},
            layout=self.layout["run_btn"],
            description="Load data",
            button_style="success",
        )
        self.run_button = create_expanded_button(
            "Run Simulation",
            "success",
            layout=self.layout["run_btn"],
            button_color=self.layout["theme_color"],
        )
        self.save_config_button = self._init_html_download_button(
            title="Save Current Network and Drives",
            mimetype="application/json",
        )
        self.save_simulation_button = self._init_html_download_button(
            title="Save Simulation Output",
            mimetype="text/csv",
        )
        # the list that corresponds to save_simulation_button
        self.simulation_list_widget = Dropdown(
            options=["Simulation Output to Save"],
            value="Simulation Output to Save",
            disabled=True,
            layout=Layout(
                width="50%",
                flex="0 1 50%",  # prevents growth beyond container limit
                min_width="0",  # forces text to truncate
            ),
        ).add_class("simulation-list-widget")

        # ==================================================
        # Network tab
        # ==================================================

        self.load_connectivity_button = FileUpload(
            accept=".json",
            multiple=False,
            style={"button_color": self.layout["theme_color"]},
            description="Load local network connectivity",
            layout=self.layout["btn_full_w"],
            button_style="success",
        )
        self.cell_type_radio_buttons = RadioButtons(
            options=["L2/3 Pyramidal", "L5 Pyramidal"], description="Cell type:"
        )
        self.cell_layer_radio_buttons = RadioButtons(
            options=["Geometry", "Synapses", "Biophysics"],
            description="Cell Properties:",
        )

        # instantiate empty list/dicts for storing network-related data
        # --------------------------------------------------
        # Connectivity tab
        self.global_gain_widgets = dict()
        self.connectivity_widgets = list()

        # Cell parameters tab
        self.cell_parameters_widgets = dict()

        # ==================================================
        # External drives tab
        # ==================================================

        # primary ("fixed") external drives tab buttons
        # --------------------------------------------------
        self.load_drives_button = FileUpload(
            accept=".json",
            multiple=False,
            style={"button_color": self.layout["theme_color"]},
            description="Load external drives",
            layout=self.layout["btn"],
            button_style="success",
        )
        self.add_drive_button = create_expanded_button(
            "Add drive",
            "primary",
            layout=self.layout["btn"],
            button_color=self.layout["theme_color"],
        )
        self.delete_drive_button = create_expanded_button(
            "Delete all drives",
            "success",
            layout=self.layout["btn"],
            button_color=self.layout["theme_color"],
        )

        # drive selection dropdown fields
        # --------------------------------------------------
        self.widget_drive_type_selection = Dropdown(
            options=["Evoked", "Poisson", "Rhythmic", "Tonic"],
            value="Evoked",
            description="Drive type:",
            disabled=False,
            layout=Layout(width="auto"),
            style={"description_width": "100px"},
        ).add_class("drive-selection")
        self.widget_location_selection = Dropdown(
            options=["Proximal", "Distal"],
            value="Proximal",
            description="Drive location:",
            disabled=False,
            layout=Layout(width="auto"),
            style={"description_width": "100px"},
        ).add_class("drive-location")

        # instantiate empty lists/widgets for storing drive-related data
        # --------------------------------------------------
        self.drive_widgets = list()
        self.drive_boxes = list()
        self.drive_accordion = Accordion()

        # ==================================================
        # Visualization tab
        # ==================================================

        # instantiate empty dictionaries for storing visualization-related data
        # --------------------------------------------------
        self.plot_outputs_dict = dict()
        self.plot_dropdown_types_dict = dict()
        self.plot_sim_selections_dict = dict()

        # ----------------------------------------------------------------------
        # Run initialization functions
        # ----------------------------------------------------------------------
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

    def _init_html_download_button(
        self,
        title,
        mimetype,
    ):
        b64 = base64.b64encode("".encode())
        payload = b64.decode()
        # Initialliting HTML code for download button
        self.html_download_button = """
        <a download="{filename}" href="data:{mimetype};base64,{payload}"
          download>
        <button
            style="background:{color_theme}; height:{btn_height}; width:{btn_width}"
            class="jupyter-button mod-warning" {is_disabled}
        >
            {title}
        </button>
        </a>
        """
        # Create widget wrapper
        return HTML(
            self.html_download_button.format(
                payload=payload,
                filename={""},
                is_disabled="disabled",
                btn_height=self.layout["run_btn"].height,
                btn_width=self.layout["save_btn"].width,
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
        self._drives_out = Output().add_class("external-drives-widgets")
        self._connectivity_out = Output().add_class("connectivity-weights-widgets")
        self._cell_params_out = Output().add_class("cell-parameters-widgets")
        self._global_gain_out = Output().add_class("connectivity-gains-widgets")

        self._log_out = Output()

        self.viz_manager = _VizManager(self.data, self.layout, self.fig_default_params)

        # detailed configuration of backends
        self._backend_config_out = Output().add_class("backend-config-out")

        # static parts
        # Running status
        self._simulation_status_bar = HTML(
            value=self._simulation_status_contents["not_running"]
        )

        # log window with toggle
        # --------------------------------------------------
        # toggle button
        self._log_toggle_btn = Button(
            icon="chevron-down",
            layout=Layout(width="30px", height="30px"),
            tooltip="Toggle Log View",
        ).add_class("log-toggle-icon")

        # log window
        self._log_window = HBox(
            [self._log_out, self._log_toggle_btn],
            layout=self.layout["log_window"],
        ).add_class("log-window")

        # store the expanded height *directly* for use in toggle_logs() below
        self._log_expanded_height = self.layout["log_window"].height

        # function to toggle log window height
        def toggle_logs(_):
            if self._log_window.layout.height == "3.5em":
                self._log_window.layout.height = self._log_expanded_height
                self._log_toggle_btn.icon = "chevron-down"
            else:
                self._log_window.layout.height = "3.5em"
                self._log_toggle_btn.icon = "chevron-up"

        # apply function when button is clicked
        self._log_toggle_btn.on_click(toggle_logs)

        # title
        sun_icon = (
            "M361.5 1.2c5 2.1 8.6 6.6 9.6 11.9L391 121l107.9 19.8c5.3 1 9.8 4.6 11.9 "
            "9.6s1.5 10.7-1.6 15.2L446.9 256l62.3 90.3c3.1 4.5 3.7 10.2 1.6 15.2s-6.6 "
            "8.6-11.9 9.6L391 391 371.1 498.9c-1 5.3-4.6 9.8-9.6 11.9s-10.7 1.5-15.2-"
            "1.6L256 446.9l-90.3 62.3c-4.5 3.1-10.2 3.7-15.2 1.6s-8.6-6.6-9.6-11.9L121 "
            "391 13.1 371.1c-5.3-1-9.8-4.6-11.9-9.6s-1.5-10.7 1.6-15.2L65.1 256 2.8 "
            "165.7c-3.1-4.5-3.7-10.2-1.6-15.2s6.6-8.6 11.9-9.6L121 121 140.9 13.1c1-"
            "5.3 4.6-9.8 9.6-11.9s10.7-1.5 15.2 1.6L256 65.1 346.3 2.8c4.5-3.1 10.2-"
            "3.7 15.2-1.6zM160 256a96 96 0 1 1 192 0 96 96 0 1 1 -192 0zm224 0a128 128 "
            "0 1 0 -256 0 128 128 0 1 0 256 0z"
        )

        moon_icon = (
            "M223.5 32C100 32 0 132.3 0 256S100 480 223.5 480c60.6 0 115.5-24.2 155.8-"
            "63.4c5-4.9 6.3-12.5 3.1-18.7s-10.1-9.7-17-8.5c-9.8 1.7-19.8 2.6-30.1 2.6c"
            "-96.9 0-175.5-78.8-175.5-176c0-65.8 36-123.1 89.3-153.3c6.1-3.5 9.2-10.5 "
            "7.7-17.3s-7.3-11.9-14.3-12.5c-6.3-.5-12.6-.8-19-.8z"
        )

        toggle_script = (
            "const c = document.querySelector('.jupyter-widgets-view') || "
            "document.body; if (c) { c.classList.toggle('dark-mode'); "
            "const isD = c.classList.contains('dark-mode'); "
            "const s = document.getElementById('sun-svg'); "
            "const m = document.getElementById('moon-svg'); "
            "s.style.display = isD ? 'none' : 'block'; "
            "m.style.display = isD ? 'block' : 'none'; }"
        )

        self._header = HTML(
            value=f"""
                <div class="title-bar-contents" style='
                    background:{self.layout["theme_color"]};
                    text-align:center;
                    color:white;
                    position:relative;
                    height: 28px;
                    line-height: 28px;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                '>
                    HUMAN NEOCORTICAL NEUROSOLVER
                    <div id="theme-toggle-container" style="
                        position: absolute;
                        left: 8px;
                        top: 0;
                        padding: 4px 0 4px 0px;
                        height: 20px;
                        display: flex;
                        align-items: center;
                        cursor: pointer;
                    ">
                        <div style="width: 20px; height: 20px; display: flex;"
                            onclick="(function() {{ {toggle_script} }})()">
                            <svg id="sun-svg" viewBox="0 0 512 512" style="
                                fill: white;
                                display: block;
                                width: 100%;
                                height: 100%;
                            ">
                                <path d="{sun_icon}"></path>
                            </svg>
                            <svg id="moon-svg" viewBox="0 0 384 512" style="
                                fill: white;
                                display: none;
                                width: 100%;
                                height: 100%;
                            ">
                                <path d="{moon_icon}"></path>
                            </svg>
                        </div>
                    </div>
                </div>
            """,
        )

    @property
    def analysis_config(self):
        """Provides everything viz window needs except for the data."""
        return {
            "viz_style": self.layout["visualization_output_figsize"],
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

        def _simulation_list_change(value):
            # Simulation Data
            _simulation_data, file_extension = _serialize_simulation(
                self._log_out, self.data, self.simulation_list_widget
            )

            self.simulation_list_widget.disabled = False

            result_file = f"{value.new}{file_extension}"
            if file_extension == ".csv":
                b64 = base64.b64encode(_simulation_data.encode())
            else:
                b64 = base64.b64encode(_simulation_data)

            payload = b64.decode()

            # redraw button in the same way after simulation change, but mapped to
            # the result_file filename
            self.save_simulation_button.value = self.html_download_button.format(
                payload=payload,
                filename=result_file,
                is_disabled="",
                btn_height=self.layout["save_btn"].height,
                btn_width=self.layout["save_btn"].width,
                color_theme=self.layout["theme_color"],
                title="Save Simulation Output",
                mimetype="text/csv",
            )

            # Network Configuration
            network_config = _serialize_config(
                self._log_out, self.data, self.simulation_list_widget
            )
            b64_net = base64.b64encode(network_config.encode())

            # redraw button in the same way after simulation change, but mapped to
            # the value.new filename
            self.save_config_button.value = self.html_download_button.format(
                payload=b64_net.decode(),
                filename=f"{value.new}.json",
                is_disabled="",
                btn_height=self.layout["save_btn"].height,
                btn_width=self.layout["save_btn"].width,
                color_theme=self.layout["theme_color"],
                title="Save Current Network and Drives",
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
        self.load_data_button.observe(_on_upload_data, names="value")
        self.simulation_list_widget.observe(_simulation_list_change, "value")
        self.widget_drive_type_selection.observe(_driver_type_change, "value")

        self.cell_type_radio_buttons.observe(_cell_type_radio_change, "value")
        self.cell_layer_radio_buttons.observe(_cell_layer_radio_change, "value")

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

    def build_sim_tab_contents(self):
        """Build the Simulation tab contents"""

        simulation_box = VBox(
            [
                HTML("Simulation Parameters").add_class("sim-tab-titles"),
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
                # the dynamic-spacer can shrink/grow based on available space to
                # help prevent any "smushing" or overlap that may appear on some
                # OS/browser combinations but not others
                Box().add_class("dynamic-spacer"),
                HTML("Default Visualization Parameters").add_class("sim-tab-titles"),
                VBox(
                    [
                        self.widget_default_smoothing,
                        self.widget_default_scaling,
                        self.widget_min_frequency,
                        self.widget_max_frequency,
                    ]
                ),
                Box().add_class("dynamic-spacer"),
                # the VBox below contains the run, save, and load buttons, as well as
                # the dropdown widget for selecting networks/simulations to save
                VBox(
                    [
                        HBox(
                            [
                                self.run_button,
                                self.load_data_button,
                            ]
                        ),
                        HBox(
                            [
                                self.save_config_button,
                            ]
                        ),
                        HBox(
                            [
                                self.save_simulation_button,
                                self.simulation_list_widget,
                            ],
                            layout=Layout(align_items="center"),
                        ),
                    ],
                    layout=Layout(
                        # don't grow, *do* allow shrink, use auto height
                        flex="0 1 auto",
                    ),
                ).add_class("sim-tab-buttons"),
            ],
            layout=self.layout["simulation_tab_contents"],
        ).add_class("simulation-tab-contents")

        return simulation_box

    def build_network_tab_contents(self):
        """build the Network tab contents"""

        network_tab_contents = Tab().add_class("network-tab-contents")

        # build Connectivity tab contents
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
                # the toggle widgets for the weights for each connection
                self._connectivity_out,
            ]
        ).add_class("connectivity-params")

        # build Cell parameters tab contents
        cell_parameters_box = VBox(
            [
                HBox([self.cell_type_radio_buttons, self.cell_layer_radio_buttons]),
                self._cell_params_out,
            ]
        ).add_class("cell-params")

        # build the Network tab from its constituent components
        network_tab_contents.children = [
            connectivity_box,
            cell_parameters_box,
        ]
        # label the sub tabs
        network_tab_contents.titles = [
            "Connectivity",
            "Cell parameters",
        ]

        return network_tab_contents

    def build_drive_tab_contents(self):
        """build the External Drives tab contents"""

        drive_load_delete_container = VBox(
            [
                self.load_drives_button,
                self.delete_drive_button,
            ],
            layout=Layout(flex="1"),
        )

        drive_add_container = VBox(
            [
                self.add_drive_button,
                self.widget_drive_type_selection,
                self.widget_location_selection,
            ],
            layout=Layout(flex="1"),
        )

        # container to hold all contents of the External Drives tab
        drive_tab_contents = VBox(
            [
                # buttons / input fields for managing drives
                HBox(
                    [
                        drive_load_delete_container,
                        drive_add_container,
                    ]
                ),
                # the toggle widgets for each of the individual drives
                self._drives_out,
            ]
        ).add_class("drive-tab-contents")

        return drive_tab_contents

    def build_parameters_window(self):
        """
        build parameters-window (to occupy AppLayout's left_sidebar)
        """

        # initialize the Vbox objects that contain the contents of the primary GUI
        # tabs: Simulation, Network, External Drives, and Visualization
        simulation_tab_contents = self.build_sim_tab_contents()
        network_tab_contents = self.build_network_tab_contents()
        drive_tab_contents = self.build_drive_tab_contents()
        visualization_tab_contents = self.viz_manager.build_viz_tab_contents()

        # build the param_window_tabs_widget Tab() object, which holds both the
        # tab bar *and* the associated contents for each tab
        param_window_tabs_widget = Tab().add_class("param-window-tabs-widget")
        param_window_tabs_widget.layout = self.layout["param_window_tabs_widget"]

        # assign tab contents
        param_window_tabs_widget.children = [
            simulation_tab_contents,
            network_tab_contents,
            drive_tab_contents,
            visualization_tab_contents,
        ]

        # loop through tabs and set the title
        titles = (
            "Simulation",
            "Network",
            "External drives",
            "Visualization",
        )
        for idx, title in enumerate(titles):
            param_window_tabs_widget.set_title(idx, title)

        # build parameters window from constituent components
        parameters_window = VBox(
            [
                VBox(
                    [param_window_tabs_widget],
                    layout=self.layout["param_tabs_outer_container"],
                ).add_class("param-tabs-outer-container"),
                self._log_window,
            ],
            layout=self.layout["parameters_window"],
        )

        return parameters_window

    def compose(self, return_layout=True):
        """Build the GUI and its widgets

        Parameters
        ----------
        return_layout : bool
            If the method returns the layout object which can be rendered by
            IPython.display.display() method.
        """

        # setup
        # --------------------------------------------------

        # add custom CSS and JS to the DOM before AppLayout is called so that
        # the style is applied before the widget is rendered
        self.custom_css_styling()

        # handle display of backend options and associated input boxes
        #   - an additional input field "MPI cmd" appears when the MPI backend is
        #     selected and disappears when "Joblib" is selected
        handle_backend_change(
            self.widget_backend_selection.value,
            self._backend_config_out,
            self.widget_mpi_cmd,
            self.widget_n_jobs,
        )

        # build parameters-window and visualization-window
        parameters_window = self.build_parameters_window()
        visualization_window = self.viz_manager.build_visualization_window()

        # build the GUI from its constituent components
        # --------------------------------------------------

        self.app_layout = AppLayout(
            header=self._header,
            left_sidebar=parameters_window,
            right_sidebar=visualization_window,
            footer=self._simulation_status_bar,
            pane_widths=[
                self.layout["parameters_window"].width,
                "0px",  # center container width, currently unused
                self.layout["visualization_window"].width,
            ],
            pane_heights=[
                self.layout["header_height"],
                self.layout["parameters_window"].height,
                self.layout["simulation_status_height"],
            ],
        ).add_class("hnn-gui")

        # add classes to the "outer-most" containers / AppLayout gridboxes
        self.app_layout.left_sidebar.add_class("parameters-window")
        self.app_layout.right_sidebar.add_class("visualization-window")
        self.app_layout.header.add_class("title-bar")
        self.app_layout.footer.add_class("status-bar")

        # initialize link callbacks to UI components
        self._link_callbacks()

        # initialize drive and connectivity ipywidgets
        self.load_drive_and_connectivity()

        if not return_layout:
            return
        else:
            return self.app_layout

    def custom_css_styling(self):
        # style the AppLayout container
        style_gui_container = HTML(
            value="""
            <style>
                .hnn-gui {
                    /*
                        center the Applayout grid on the webpage. The addition of the
                        "safe" parameter prevents the app from being cut off when the
                        window size is smaller than the gui itself
                    */
                    justify-content: safe center !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(style_gui_container)

        # add styling to children of param-window-tabs-widget
        param_tabs_styling = HTML(
            value="""
            <style>
                /*
                    ensure the border around the container that holds the tabs
                    themselves takes up no space, i.e. 0px
                */
                .param-window-tabs-widget .widget-tab-bar {
                    border: 0px solid lightgrey;
                }

                /* set the border color for the individual tabs */
                .param-window-tabs-widget .lm-TabBar-tab {
                    border-color: lightgrey !important;
                }

                /*
                    this adjusts the content container's border, not the borders
                    on the tabs themselves. Though it does effectively set the
                    bottom border around the inactive tabs (with the active tab
                    not having a bottom border)
                */
                .param-window-tabs-widget .widget-tab-contents {
                    border: 1px solid lightgrey !important;
                }

                /*
                    allow TabBar to grow to fill the available space; this fixes
                    and issue where the last tab was slightly offset (by around
                    1px) from the right border. Not that we only target tabs that
                    are the first child of .lm-TabBar-content to ensure none of
                    the nested tabs grow
                */
                .param-window-tabs-widget > .widget-tab-bar >
                .lm-TabBar-content > .lm-TabBar-tab {
                    flex-grow: 1 !important;
                }

            /*
                adjust bottom-padding around widget-tab-contents

                bonus: optionally target a single tab (e.g., the first tab) with...
                .param-window-tabs-widget >
                .widget-tab-bar:has(.lm-TabBar-tab:nth-child(1).lm-mod-current) +
                .widget-tab-contents
            */
            .param-window-tabs-widget >
            .widget-tab-contents {
                padding-bottom: 10px !important;
            }


            </style>
            """,
            layout=Layout(display="none"),
        )
        display(param_tabs_styling)

        # make subtabs (e.g., the Connectivity tab under Network) "sticky"
        make_subtabs_sticky = HTML(
            value="""
            <style>
                .network-tab-contents {
                    height: 99% !important;
                    flex: none !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(make_subtabs_sticky)

        # disable dropdown menu displaying when no actual items are present
        # note: i've *only* noticed this on Firefox, but it creates an empty
        #       oval on screen that looks like an erroneous box... this requires
        #       js as you can't target those popup boxes with CSS, unfortunately

        js_code = """
            (function() {
                        const blockEmpty = (e) => {
                            const t = e.target;
                            const isDrop = t.closest('.widget-dropdown');
                            if (t.tagName === 'SELECT' && isDrop) {
                                if (t.childElementCount === 0) {
                                    e.preventDefault();
                                    t.focus();
                                }
                            }
                        };

                        document.addEventListener('mousedown', blockEmpty, true);

                        const obs = new MutationObserver(() => {
                            document.removeEventListener('mousedown', blockEmpty, true);
                            document.addEventListener('mousedown', blockEmpty, true);
                        });

                        obs.observe(document.body, {childList: true, subtree: true});
            })();
            """

        display(Javascript(js_code))

        # adjust colors and accents
        adjust_accent_colors = HTML(
            value="""
                <style>
                    /* set or overwrite root variables */
                    :root {
                        /* set color variables */
                        --theme-color: #802989;
                        --textbook-light-purple: #ba83be;
                        --textbook-sidebar-purple: #88548c;
                        --default-blue-accent: #64b5f6;
                        --gentle-red: #ed665e;
                        --gentle-green: #77aa77;
                        --statusbar-running: orange;

                        /* adjust border colors around input fields */
                        # --jp-widgets-input-focus-border-color: var(
                        #     --default-blue-accent
                        # ) !important;

                        --jp-widgets-input-focus-border-color: var(
                            --textbook-light-purple
                        ) !important;

                        --jp-error-color1: var(--gentle-red) !important;
                    }

                    /* adjust the accent line above the selected tab */
                    .lm-TabBar-tab.lm-mod-current::before {
                        background-color: var(--theme-color) !important;
                    }

                    /* adjust border width on focus for input fields */
                    .jupyter-widgets input:focus,
                    .jupyter-widgets select:focus,
                    .jupyter-widgets textarea:focus {
                        --jp-widgets-input-border-width: 2px !important;

                        /*  change border-width property directly to force update.  */
                        border-width: var(--jp-widgets-input-border-width) !important;

                        /*
                            adjust text padding on focus to account for increase
                            in border thickness from 1px to 2px
                        */
                        padding:
                            var(--jp-widgets-input-padding)
                            0
                            var(--jp-widgets-input-padding)
                            calc(var(--jp-widgets-input-padding) * 2 - 1px)
                            !important;
                    }

                </style>
            """,
            layout=Layout(display="none"),
        )
        display(adjust_accent_colors)

        log_toggle = HTML(
            value="""
            <style>
                .log-toggle-icon {
                    overflow: visible !important;
                    position: absolute !important;
                    top: 0px !important;
                    left: 0px !important;
                    z-index: 10;
                    color: var(--textbook-light-purple);
                    background: transparent !important;
                    border: none !important;
                }

                /*
                    ensure the log output handles scrolling,
                    and that the output content doesn't overlap the button
                */
                .log-window > .widget-output {
                    overflow-y: auto !important;
                    padding-left: 30px !important;
                    padding-right: 10px !important;
                }

                /* remove focus and hover effects */
                .log-toggle-icon:focus {
                    outline: none !important;
                    border: none !important;
                    box-shadow: none !important;
                    --jp-widgets-input-focus-border-color: transparent !important;
                    --jp-widgets-input-focus-shadow: none !important;
                }

                .log-toggle-icon:hover {
                    color: var(--theme-color);
                    box-shadow: none !important;
                    # transform: scale(1.2) !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(log_toggle)

        tabs_add_scrollbar_gutter = HTML(
            value="""
            <style>
                /*
                    identify the tab bar
                    check if its 3rd tab (external drives) is active
                    selectively style the contents container for that tab
                */
                .param-window-tabs-widget >
                .widget-tab-bar:has(.lm-TabBar-tab:nth-child(3).lm-mod-current) +
                .widget-tab-contents {
                    scrollbar-gutter: stable !important;
                    overflow-y: auto !important;
                    padding-right: 10px !important;
                }

                /* same for 4th tab (visualization) */
                .param-window-tabs-widget >
                .widget-tab-bar:has(.lm-TabBar-tab:nth-child(4).lm-mod-current) +
                .widget-tab-contents {
                    scrollbar-gutter: stable !important;
                    overflow-y: auto !important;
                    padding-right: 10px !important;
                }

                .network-tab-contents > .widget-tab-contents {
                    scrollbar-gutter: stable !important;
                    overflow-y: auto !important;
                    padding-right: 10px !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(tabs_add_scrollbar_gutter)

        stabilize_tabs_height = HTML(
            value="""
            <style>
                /* prevent small pixel shifts when switching between tabs */
                .jupyter-widget-TabPanel > .widget-tab-contents {
                    /*
                        force the tab parent container to use a 'fixed' flex-basis
                        instead of 'auto', ensuring sizes are calculate based on the
                        parent's dimensions rather than the child's content. this
                        prevents height recalculation when the layout changes (e.g.,
                        when a new scrollbar appears), which can otherwise cause
                        visible pixel 'jumps'
                    */
                    flex-basis: 100% !important;

                    /*
                        ensure borders and padding are contained within the height
                        to prevent shifts when the layout changes
                    */
                    box-sizing: border-box !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(stabilize_tabs_height)

        adjust_viz_window_spacing = HTML(
            value="""
                <style>

                    /* adjust the tab border color to match the container */
                    .visualization-window .lm-TabBar-tab {
                        border-color: lightgrey !important;
                    }

                    /* remove the automatically-added margins */
                    .visualization-window > .widget-output,
                    .visualization-window .fig-tabs {
                        margin: 0px 0px 0px 0px !important;
                    }

                    /*
                        make the visualization-window border transparent if
                        fig-tabs is not empty
                    */
                    .visualization-window:has(.fig-tabs:not(:empty)) {
                        border-color: transparent !important;
                    }

                    /*
                        align TabBar to the *top* rather than the bottom, and shift
                        the top margin to reduce excess space
                    */
                    .visualization-window .lm-TabBar {
                        align-items: flex-start !important;
                        margin-top: -1px !important;
                    }

                    .visualization-window .lm-TabBar-content {
                        margin-top: 0 !important;
                    }

                    /*
                        set 'display: flex' on visualization-window to enable its
                        children to grow to fill available space
                    */
                    .visualization-window {
                        display: flex !important;
                        flex-direction: column !important;
                    }

                    /*
                        target child elements inside visualization window to allow
                        them to grow to fill the available space. This looks more
                        complicated than it actually is, we just need to target
                        all of the children manually, which necessitates understanding
                        the 'hierarchy' of automatically-generated containers
                    */
                    .visualization-window > .widget-output,
                    .visualization-window > .widget-output > .jp-OutputArea,
                    .visualization-window .jp-OutputArea > .jp-OutputArea-child,
                    .visualization-window .jp-OutputArea-child > .jp-OutputArea-output,
                    .visualization-window .jp-OutputArea-output > .fig-tabs,
                    .visualization-window .widget-tab-contents {
                        display: flex !important;
                        flex-direction: column !important;
                        flex: 1 1 0% !important;
                        height: auto !important;
                        min-height: 0 !important;
                    }

                    /* style the tab container that holds the output */
                    .visualization-window .widget-tab-contents {
                        border: 1px solid lightgrey !important;
                        box-sizing: border-box !important;
                        margin: 0px !important;
                        width: 100% !important;
                    }

                    /* fix interactive matplotlib plot sizing issues */
                    /* --------------------------------------------- */
                    /*
                        collapse the actual height and force the container to
                        have height via padding; this change is necessary to force
                        the interactive plots to respect the parent container's
                        constraints
                    */
                    .visualization-window .jupyter-matplotlib-canvas-div {
                        width: 100% !important;
                        max-width: 100% !important;
                        height: 0 !important;
                        padding-bottom: 100% !important;
                        position: relative !important;
                        display: block !important;
                    }

                    /* scale both canvases to fit that padded container */
                    .visualization-window .jupyter-matplotlib-canvas-div canvas {
                        width: 100% !important;
                        height: 100% !important;
                        position: absolute !important;
                        top: 0 !important;
                        left: 0 !important;
                    }

                    /* manually set figure dimensions so the canvas actually renders */
                    .visualization-window .jupyter-matplotlib,
                    .visualization-window .jupyter-matplotlib-figure,
                    .visualization-window .jupyter-matplotlib-canvas-container {
                        width: 650px !important;
                        height: auto !important;
                        min-height: 0 !important;
                        overflow: visible !important;
                    }
                </style>
            """,
            layout=Layout(display="none"),
        )
        display(adjust_viz_window_spacing)

        adjust_fig_placeholer = HTML(
            value="""
            <style>
                .fig-placeholder .widget-html-content {
                    color: #969696;
                    font-size: 13px;
                    margin-top: 6px;
                    text-align: center;
                }
            </style>
            """
        )
        display(adjust_fig_placeholer)

        adjust_viz_param_tab = HTML(
            value="""
            <style>
                /*
                    remove the rectangle in the visualization-tab when there is
                    not actually any figure present
                */

                /*
                    remove the "top" edge of the rectangle by setting border to none if
                    the "lm-TabBar-tab" child class is not present

                    this element needs to be targeted with "::after", as that element
                    is used to form the border that spans from the end of the tab to
                    the end of the container that holds the figure
                */
                .visualization-tab
                .lm-TabBar-content:not(:has(.lm-TabBar-tab))::after {
                    content: none !important;
                    border: none !important;
                    display: none !important;
                }

                /*
                    remove the "U" edges of the rectangle by setting border to none if
                    the "widget-container" child is not present
                */
                .visualization-tab
                .widget-tab-contents:not(:has(.widget-container)) {
                    border: none !important;
                    box-shadow: none !important;
                    outline: none !important;
                }

            </style>
            """,
            layout=Layout(display="none"),
        )
        display(adjust_viz_param_tab)

        adjust_tab_overflow = HTML(
            value="""
            <style>

            :root {
                --tab-color: white;
                --tab-border: lightgrey;
            }

            /*
                target the TabBar in visualization-window (right panel)
                target the *first* TabBar instance in visualization-tab (left panel)
            */
            .visualization-window .lm-TabBar,
            .visualization-tab .lm-TabBar:first-of-type {
                position: relative !important;
                border-bottom: none !important;
            }

            /*
                for both visualization-window and visualization-tab:
                    - set TabBar container properties
                    - remove the bottom border, as we'll be managing it via the
                      individual tabs instead
                    - allow content to overflow horizontal axis
            */
            .visualization-window .lm-TabBar-content,
            .visualization-tab .lm-TabBar:first-of-type .lm-TabBar-content {
                max-width: 100% !important;
                overflow-x: auto !important;
                scrollbar-width: none !important;
                display: flex !important;
                -ms-overflow-style: none !important; /* old browser compatibility */
                border-bottom: none !important;
                padding-top: 1px !important;
            }

            /*
                hide the scrollabr for both visualization containers
            */
            .visualization-window .lm-TabBar-content::-webkit-scrollbar,
            .visualization-tab .lm-TabBar:first-of-type
            .lm-TabBar-content::-webkit-scrollbar {
                display: none !important;
            }

            /*
                for both visualization-window and visualization-tab:
                    - style the individual tabs
                    - prevent the individual tabs from shrinking
            */
            .visualization-window .lm-TabBar-tab,
            .visualization-tab .lm-TabBar:first-of-type .lm-TabBar-tab {
                flex-shrink: 0 !important;
                border-bottom: 1px solid var(--tab-border) !important;
                margin-top: 0px !important;
                transform: none !important;
            }

            /*
                for both visualization-window and visualization-tab, we set the
                bottom border of the currently-selected tab to equal the background
                color of the tab
            */
            .visualization-window .lm-TabBar-tab.lm-mod-current,
            .visualization-tab .lm-TabBar:first-of-type .lm-TabBar-tab.lm-mod-current {
                border-bottom: 1px solid var(--tab-color) !important;
                /*
                    background-color does not strictly need to be specified here, but
                    keeping it as a fail safe
                */
                background-color: var(--tab-color) !important;
                margin-top: 0px !important;
            }

            /*
                for both visualization-window and visualization-tab, we need to
                set the bottom border in areas where there is *not* a tab (i.e.,
                there are too few tabs to fill up the entire horizontal width). The
                border needs to grow to fill the available space between the tabs
                and the end of the container.
            */
            .visualization-window .lm-TabBar-content::after,
            .visualization-tab .lm-TabBar:first-of-type .lm-TabBar-content::after {
                content: '';
                flex-grow: 1;
                border-bottom: 1px solid var(--tab-border) !important;
                margin-bottom: 0px;
            }

            /* top borders are managed by the tabs for both visualization containers */
            .visualization-window .widget-tab-contents,
            .visualization-tab .widget-tab-contents {
                border-top: none !important;
            }

            /* ----- caret styling and placement ------ */
            /* ---------------------------------------- */
            .tab-caret {
                position: absolute;
                top: 1px;
                bottom: 0;
                margin: auto 0;
                height: 70%;
                width: 20px;
                border-radius: 20px;
                background: #ffffffcc;
                color: var(--textbook-light-purple) !important;
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 30 !important;
                cursor: pointer !important;
                visibility: hidden;
                border: none;
                user-select: none;
                pointer-events: auto !important;
            }

            .tab-caret.is-visible { visibility: visible; }
            .caret-left { left: 4px; }
            .caret-right { right: 4px; }

            </style>

            /* ----------------------------------------------------------------------
                this method of running the script below seems a bit "hacky", but it is
                apparently a commonly used trick. We use a 1x1 transparent GIF to
                trigger the "onload" event that follows it. this is apparently the
                smallest footprint option for a valid asset load, meaning the browser
                won't treat it as an error. this method ensure the script runs every
                time a tab is rendered.
               ----------------------------------------------------------------------
            */
            <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///
yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
                onload='
                (function() {
                    const setup = (selector) => {
                        const parent = document.querySelector(selector);
                        if (!parent) return;

                        const bar = parent.querySelector(".lm-TabBar-content");
                        if (!bar || parent.querySelector(".caret-left")) return;

                        const lCaret = document.createElement("div");
                        lCaret.className = "tab-caret caret-left";
                        lCaret.innerHTML = "&#10094;";
                        const rCaret = document.createElement("div");
                        rCaret.className = "tab-caret caret-right";
                        rCaret.innerHTML = "&#10095;";

                        const leftWall = document.createElement("div");
                        const rightWall = document.createElement("div");

                        let wallStyle = "position: absolute; top: 2px; ";
                        wallStyle += "bottom: 0; width: 1px; ";
                        wallStyle += "background-color: var(--tab-border); ";
                        wallStyle += "z-index: 20; ";
                        wallStyle += "pointer-events: none; display: none;";

                        leftWall.style.cssText = wallStyle + "left: 0;";
                        rightWall.style.cssText = wallStyle + "right: 0;";

                        parent.appendChild(lCaret);
                        parent.appendChild(rCaret);
                        parent.appendChild(leftWall);
                        parent.appendChild(rightWall);

                        const update = () => {
                            const canLeft = bar.scrollLeft > 1;
                            const isAtEnd = bar.scrollLeft +
                                bar.clientWidth >= (bar.scrollWidth - 1);

                            lCaret.classList.toggle("is-visible", canLeft);
                            rCaret.classList.toggle("is-visible", !isAtEnd);

                            leftWall.style.display = canLeft ? "block" : "none";
                            rightWall.style.display = !isAtEnd ? "block" : "none";
                        };

                        const doScroll = (e, amt) => {
                            e.stopImmediatePropagation();
                            e.stopPropagation();
                            e.preventDefault();
                            bar.scrollBy({ left: amt, behavior: "smooth" });
                            return false;
                        };

                        const events = ["pointerdown", "mousedown", "click"];
                        events.forEach(evtName => {
                            lCaret.addEventListener(evtName, (e) =>
                                doScroll(e, -150), true);
                            rCaret.addEventListener(evtName, (e) =>
                                doScroll(e, 150), true);
                        });

                        bar.addEventListener("scroll", update);
                        const obs = new MutationObserver(update);
                        obs.observe(bar, { childList: true, subtree: true });
                        setTimeout(update, 100);
                    };

                    const poller = setInterval(() => {
                        setup(".visualization-window .lm-TabBar");
                        setup(".visualization-tab .lm-TabBar:first-of-type");
                    }, 500);
                })();
            '>
            """,
            layout=Layout(display="none"),
        )
        display(adjust_tab_overflow)

        adjust_header_footer_margins = HTML(
            value="""
            <style>
                .title-bar {
                    margin: 2px 0px 2px 0px !important;
                }
                .title-bar-contents {
                    margin: 0px !important;
                }
                .status-bar {
                    margin: 2px 0px 2px 0px !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(adjust_header_footer_margins)

        # viz_manager relies on the presence of the "existing-plots" HTML container
        # in various parts of the code; however, the display for existing-plots
        # is redundant and adds clutter to visualization window. Since its not
        # obvious how to untangle viz_manager's dependence on this element (though
        # we should do so in a new PR...), we can simply hide the widget from the user.
        hide_existing_plots = HTML(
            value="""
            <style>
                /* hide the existing-plots container */
                .jupyter-widgets.widget-vbox.existing-plots {
                    display: none !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(hide_existing_plots)

        # adjust carets on dropdown input fields
        adjust_dropdown_carets = HTML(
            value=(
                """
                <style>
                /*
                    remove the default jupyter caret from the dropdown field,
                    and style the input field for longer strings
                */
                .widget-dropdown select,
                .widget-dropdown select:focus {
                    background-image: none !important;
                    appearance: none;
                    -webkit-appearance: none;
                    padding-right: 32px !important;
                    text-overflow: ellipsis !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                }

                /* custom caret */
                .widget-dropdown::after {
                    content: "";
                    position: absolute;
                    right: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 20px;
                    height: 20px;
                    pointer-events: none;
                    background-color: var(--jp-widgets-input-color) !important;

                    /* splitting long strings for ruff */
                    -webkit-mask-image: url('data:image/svg+xml;utf8,<svg """
                """xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">"""
                """<path d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z"/>"""
                """</svg>');
                        mask-image: url('data:image/svg+xml;utf8,<svg """
                """xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">"""
                """<path d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z"/>"""
                """</svg>');

                    -webkit-mask-repeat: no-repeat;
                    mask-repeat: no-repeat;
                    -webkit-mask-size: contain;
                    mask-size: contain;
                }

                /* recolor caret on hover/focus */
                .widget-dropdown:hover::after {
                    background-color: #ba83be !important;
                }
                .widget-dropdown:focus-within::after {
                    background-color: #ba83be !important;
                }

                /* recolor hover/focus differently when input is disabled */
                .widget-dropdown:has(select[disabled])::after {
                    background-color: grey !important;
                }
                </style>
                """
            ),
            layout=Layout(display="none"),
        )
        display(adjust_dropdown_carets)

        sim_tab_adjustments = HTML(
            value="""
            <style>
                /*
                    style the text titles differentiating run parameters from
                    default visualization parameters
                */

                .sim-tab-titles {
                    background: gray;
                    color: white;
                    width: 300px;
                    padding: 0px 5px;
                    margin-bottom: 2px;
                }

                /*
                    allow the space between sections to dynamically shrink/grow
                    based on available space so elements don't overlap
                */

                .dynamic-spacer {
                    flex: 1 1 auto !important;
                    min-height: 4px !important;
                    max-height: 15px !important;
                }

                /* add left margin to the dropdown widget for saving outputs */
                .simulation-list-widget {
                    margin-left: 6px !important;
                }

                /* adjust sim tab button properties to avoid overlap or overflow */
                .simulation-tab-contents,
                .simulation-tab-contents .widget-hbox,
                .simulation-tab-contents .widget-vbox {
                    overflow: hidden !important;
                    box-sizing: border-box !important;
                }

                /*
                    there is an additional Output() box around "Core" and "MPI cmd"
                    since the "MPI cmd" is dynamically inserted when the MPI backend
                    is selected. We need to remove the margin around that extra box
                    so that the input fields line up properly
                */
                .backend-config-out {
                    margin: 0px !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(sim_tab_adjustments)

        make_fig_button = HTML(
            value="""
            <style>
                .make-fig-btn {
                    flex: 1 1 auto !important;
                    width: auto !important;
                    margin: 2px 2px 2px 10px !important;
                }
            </style>
            """,
            layout=Layout(display="none"),
        )
        display(make_fig_button)

        dark_theme = HTML(
            value="""
            <style>
                /* basic dark-mode colors */
                .dark-mode {
                    --dm-bg-primary: #14181e;
                    --dm-bg-secondary: #313438;
                    --dm-text-main: #dfdfdf;
                    --dm-border-color: #3e3e42;
                    --dm-theme: #ba83be;  /* note: same as textbook-light-purple */
                    --statusbar-running: #C88809;
                }

                /* overwrite light-mode variables for dark-mode */
                .dark-mode {
                    --tab-color: var(--dm-bg-primary);
                    --tab-border: var(--dm-border-color);
                }

                /* force background color for major containers */
                .dark-mode,
                .dark-mode .jp-Notebook,
                .dark-mode .lm-Widget,
                .dark-mode .lm-Panel,
                .dark-mode .jupyter-widgets:not(.jupyter-button) {
                    background-color: var(--dm-bg-primary) !important;
                    border-color: var(--dm-border-color) !important;
                }

                /*
                    flip text and border colors globally. we'll have to "overwrite"
                    some of these changes, but this is the best way to ensure
                    everything gets updated without manually needing to specify
                    every single child element that needs to change
                */
                .dark-mode * {
                    color: var(--dm-text-main) !important;
                    border-color: var(--dm-border-color) !important;
                }

                /* ------------------------------------------- */
                /* --- add exceptions to dark mode changes --- */
                /* ------------------------------------------- */

                /* set title bar to dark mode theme */
                .dark-mode .title-bar-contents {
                    background-color: var(--dm-theme) !important;
                    /* color: var(--dm-bg-primary) !important; */
                    color: #fff !important;
                }

                /* adjust visualization window placeholder text */
                .dark-mode .fig-placeholder .widget-html-content {
                    color: #848484 !important;
                }

                /* restore transparent outer border when fig-tabs is not empty */
                .visualization-window:has(.fig-tabs:not(:empty)) {
                    border-color: transparent !important;
                }

                /* adjust style of tabbar and child tabs */
                .dark-mode .lm-TabBar-tab {
                    background-color: var(--dm-bg-secondary) !important;
                    color: var(--dm-text-main) !important;
                    border-color: var(--dm-border-color) !important;
                }

                /* adjust the accent line above the selected tab */
                .lm-TabBar-tab.lm-mod-current::before {
                    background-color: var(--dm-theme) !important;
                }

                /* adjust active tabbar to blend with primary background */
                .dark-mode .lm-TabBar-tab.lm-mod-current {
                    background-color: var(--dm-bg-primary) !important;
                }

                /* adjust tab caret that controls scrolling */
                .tab-caret {
                    color: var(--textbook-light-purple) !important;
                }

                /* adjust the caret for dropdown input fields */
                .dark-mode .widget-dropdown::after {
                    background-color: var(--dm-text-main) !important;
                }
                .dark-mode .widget-dropdown:hover::after {
                    background-color: #ba83be !important;
                }
                .dark-mode .widget-dropdown:focus-within::after {
                    background-color: #ba83be !important;
                }

                /* adjust background color for input areas */
                .dark-mode input,
                .dark-mode select,
                .dark-mode textarea {
                    background-color: var(--dm-bg-secondary) !important;
                }

                /* restore border color when focusing on input areas */
                .dark-mode input:focus,
                .dark-mode select:focus,
                .dark-mode textarea:focus {
                    border-color: var(--dm-theme) !important;
                }

                /* adjust button colors */
                .dark-mode button:not(.log-toggle-icon),
                .dark-mode .jupyter-button:not(.log-toggle-icon),
                .dark-mode .widget-button:not(.log-toggle-icon) {
                    background-color: var(--dm-theme) !important;
                    color: var(--dm-bg-primary) !important;
                }

                /* adjust button icon color */
                .dark-mode button:not(.log-toggle-icon) i,
                .dark-mode .jupyter-button:not(.log-toggle-icon) i,
                .dark-mode .widget-button:not(.log-toggle-icon) i {
                    color: var(--dm-bg-primary) !important;
                }

                /* restore toggle icon for the log */
                .dark-mode .log-toggle-icon,
                .dark-mode .log-toggle-icon i {
                    color: var(--dm-theme) !important;
                }

                /* adjust scrollbars for dark mode */
                .dark-mode ::-webkit-scrollbar {
                    width: 17px !important;
                    height: 17px !important;
                }
                .dark-mode ::-webkit-scrollbar-track {
                    background: var(--dm-bg-primary) !important;
                }
                .dark-mode ::-webkit-scrollbar-thumb {
                    background: var(--dm-bg-secondary) !important;
                    border: 4px solid var(--dm-bg-primary) !important;
                    border-radius: 20px !important;
                }
                .dark-mode ::-webkit-scrollbar-thumb:hover {
                    background: var(--dm-theme) !important;
                }

                /* dim hard-white backgrounds in visualizations */
                .dark-mode .visualization-window .widget-tab-contents img,
                .dark-mode .visualization-window .widget-tab-contents canvas,
                .dark-mode .visualization-window .widget-tab-contents .js-plotly-plot
                .main-svg,
                .dark-mode .visualization-window .widget-tab-contents .bk-root {
                    filter: brightness(0.8) contrast(1.2) !important;
                }

                /* restore red coloring for close fig and delete drive buttons */
                .dark-mode .widget-button.red-button {
                    background-color: var(--gentle-red) !important;
                }

                /* adjust param container border colors to match theme */
                /* --------------------------------------------------- */
                /*
                    Note: here I am adding accent borders to the "outer"
                    containers in dark-mode. It requires some style overwriting.
                    I think these accents make it easier to discern the "outer"-
                    most parent container from the nested inner containers. But this
                    whole section can be removed without interfering with any
                    functionality if we decide we would prefer to not have the
                    accent borders. There are three sub sections here handle:
                        - param-window-tabs-widget
                        - log-window
                        - visualization-window
                */

                /* set border for the tab contents */
                /* ------------------------------ */

                .dark-mode div.param-window-tabs-widget.param-window-tabs-widget
                > .widget-tab-contents {
                    position: relative;
                    border: 2px solid var(--dm-theme) !important;
                }

                /*
                    allow the TabBar to overflow, this will allow us to cover the
                    border created for the tab contents for the active tab
                */
                .dark-mode div.param-window-tabs-widget.param-window-tabs-widget
                > .lm-TabBar {
                    position: relative;
                    overflow: visible !important;
                }

                /* set position and borders for active tab */
                .dark-mode div.param-window-tabs-widget.param-window-tabs-widget
                > .lm-TabBar:first-of-type .lm-TabBar-tab.lm-mod-current {
                    position: relative;
                    background-color: var(--tab-color) !important;
                    border-right: 2px solid var(--dm-theme) !important;
                    border-left: 2px solid var(--dm-theme) !important;

                    /*
                        note: removing border-bottom allows the mask we create below
                        to work without creating the "triangle" shaped overlap at the
                        edges where the borders meet
                    */
                    border-bottom: none !important;
                }

                /*
                    add the "mask" that covers the bottom border on the widget-contents
                    for the active tab
                */
                .dark-mode div.param-window-tabs-widget.param-window-tabs-widget
                .lm-TabBar-tab.lm-mod-current::after {
                    content: '';
                    position: absolute;
                    bottom: -2px;
                    left: 0px;
                    right: 0px;
                    height: 2px;
                    background-color: var(--tab-color) !important;
                    z-index: 4;
                }

                /* restore colors for the simulation tab titles */
                .dark-mode .widget-html.sim-tab-titles {
                    background-color: gray !important;
                    color: white !important;

                }

                /* set border for the log window */
                /* ------------------------------ */

                .dark-mode div.log-window.log-window {
                    border: 2px solid var(--dm-theme) !important;
                }

                /* set borders for visualization window */
                /* ------------------------------ */

                /* remove top border on contents as we'll manage it with tabs */
                .dark-mode .visualization-window .widget-tab-contents {
                    border: 2px solid var(--dm-theme) !important;
                    border-top: none !important;
                }

                /* keep the 2px border-bottom for inactive tabs */
                .dark-mode .visualization-window .lm-TabBar-tab {
                    margin-top: 0px !important;
                    box-sizing: border-box !important;
                    border-bottom: 2px solid var(--dm-theme) !important;
                }

                /* style the active tab */
                .dark-mode .visualization-window .lm-TabBar-tab.lm-mod-current {
                    background-color: var(--tab-color) !important;
                    border-left: 2px solid var(--dm-theme) !important;
                    border-right: 2px solid var(--dm-theme) !important;

                    /*
                        using a box shadow "mask" instead of border-bottom fixes the
                        overlapping triangles at border edges
                    */
                    /* remove border bottom */
                    border-bottom: none !important;

                    /* add 2px padding to replace the 2px border */
                    padding-bottom: 2px !important;

                    /* draw the mask using the inset shadow */
                    box-shadow: inset 0 -2px 0 var(--tab-color) !important;
                }

                /*
                    this isn't really necessary since we've shifted the border to the
                    tabs themselves, but keeping it in as a failsafe
                */
                .dark-mode .visualization-window .lm-TabBar-content::after {
                    border-bottom: 2px solid var(--dm-theme) !important;
                }

                /* ------------------- end section ------------------- */
                /* --------------------------------------------------- */



            </style>
            """,
            layout=Layout(display="none"),
        )
        display(dark_theme)

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

    def load_drive_and_connectivity(self):
        """Add drive and connectivity ipywidgets from params."""
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
        ).add_class("red-button")
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
            else:
                raise ValueError

            print(f"Loaded {load_type} from {param_dict['name']}")
        # Resets file counter to 0
        change["owner"].set_trait("value", ([]))
        return params


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


def _get_rhythmic_widget(
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


def _get_poisson_widget(
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


def _get_evoked_widget(
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


def _get_tonic_widget(name, tstop_widget, layout, style, data=None):
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
        drive, drive_box = _get_rhythmic_widget(
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
        drive, drive_box = _get_poisson_widget(
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
        drive, drive_box = _get_evoked_widget(
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
        drive, drive_box = _get_tonic_widget(
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
    connectivity_out_style = HTML(
        """
        <style>
            /* CSS to style elements inside the Accordion */
            .connectivity-section .jupyter-widget-Collapse-contents {
                padding: 0px 0px 10px 0px !important;
                margin: 0 !important;
            }
        </style>
        """,
        layout=Layout(display="none"),
    )

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

                # 1. identify which case of global_gain_textfield applies to this
                #    src/target
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
            _exp_g_at_dist, gbar_at_zero=param_list[19].value, exp_term=3e-3, offset=0.0
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


def launch():
    """Launch voila with hnn_widget.ipynb.

    You can pass voila commandline parameters as usual.
    """
    from voila.app import main

    notebook_path = Path(__file__).parent / "hnn_widget.ipynb"
    main([str(notebook_path.resolve()), *sys.argv[1:]])
