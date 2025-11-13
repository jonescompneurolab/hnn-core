"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>
import base64
import codecs
import io
import json
import logging
import mimetypes
import re
import sys
import textwrap
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from IPython.display import IFrame, display
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
    link,
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
from hnn_core.optimization import Optimizer, generate_opt_history_table
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
        #
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
        #       tags to identify the relevant containers.
        #
        # Using our custom class names in place of the AppLayout parameter names, the
        # structure of the GUI can similarly be written as follows:
        #
        #   | ----------------- title-bar ----------------- |
        #   |   parameters-window  | | visualization-window |
        #   | ----------------- status-bar ---------------- |
        #
        #     > Note that we do not (currently) utilize the "center" AppLayout
        #       container, which is set to 0px in our AppLayout instantiation below.
        #
        # The diagrams below outline the structure of these automatically-generated
        # "outer" parent containers, with our included HTML tags:
        #
        # The AppLayout header (title-bar):
        #
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
        # The AppLayout left-sidebar (parameters-window):
        #
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
        # The AppLayout right-sidebar (visualization-window):
        #
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
        #       is initialized in _viz_manager.py, and not in gui.py .
        #
        # The AppLayout footer (status-bar):
        #
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
        # primary "outer" containers that house the actual GUI contents.

        # ----------------------------------------------------------------------
        # Set the layout properties for various GUI components
        # ----------------------------------------------------------------------

        # Set up container height / width parameters
        # ----------------------------------------------------------------------
        # Containers' properties are computed relative to total height/width,
        # allowing us to scale the GUI size without needed to figure out what the
        # exact pixel values need to be for each element:
        self.total_height = total_height
        self.total_width = total_width

        # We'll compute pixels for the "fixed" outer containers (per AppLayout), but
        # we'll be able to use percentages for most of the "inner" containers.
        # Note that we must use int() as we cannot have fractional pixel values.
        parameters_window_width = int(total_width * param_window_width_prop)
        figures_window_width = int(total_width - parameters_window_width)
        main_content_height = total_height - status_height

        # Specify the gap between the footer ("status-bar") and the containers above it
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
            # argument via "self.viz_manager = _VizManager(...)" below.
            # TODO: [DSD] this parameter should ideally be separated from self.layout,
            # which should only define the layout properties of containers.
            "dpi": dpi,
            #
            # TODO: [DSD] IMHO these elements below, which describe buttons, should
            # also be separated from self.layout. Similar to dpi, "theme_color" and
            # "btn" are passed to viz_layout, albeit they are only used in one
            # place in _viz_manager, when self.make_fig_button is called. This button
            # already has make-fig-btn HTML class, so adding the styling directly via
            # CSS to the different buttons may be the better approach.
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
            #
            # Note that "widget-tab-contents" is an auto-generated container that
            # we do not specifically tag, but it is often used in conjunction with
            # the parent or child container for targeted CSS styling.
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
            # AES
            # optimization related
            "opt_textbox": Layout(
                width="250px",
            ),
            # end of AES
            #
            # container for the log window
            #   - child of "parameters-window"
            #   - associated html class: "log-window"
            #
            # Note: we use a margin on log-window to set the footer-gap here, as the
            # gap is inserted between the inner container "log-window" and its parent
            # container "parameters-window."" Adding the gap as a margin directly to
            # parameters-window would also require recalculating the height of the
            # container to accommodate the extra margin, since its height is fixed.
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
            # Note: we set the footer-gap here by recalculating the height of
            # visualization-window. Adding footer_gap as a margin, as done above for
            # log-window above, would not shift the content up, as the container height
            # is still specified in pixels. Rather, the margin on visualization-window
            # would simply overflow into the footer.
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
            # Note: figure sizes are set in _add_figure in _viz_manager, where
            # percents are converted to pixels. This CSS block sets the dimensions
            # for both static figure outputs (<img src="data:img/png;base64,...>")
            # AND for dynamic figure outputs (<div class="jupyter-matplotlib-figure">).
            "visualization_output_figsize": Layout(
                width="100%",
                height="95%",
            ),
        }

        # Set up for the simulation status bar
        # ----------------------------------------------------------------------
        # We directly set up the html for the status bar below.
        #   - child of status-bar
        #   - associated html class: sim-status-box
        # Note: This dict is referenced in _init_ui_components and run_button_clicked:
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
            "opt_running": """
                <div
                class='sim-status-box status-running'
                style='
                    background:var(--statusbar-running);
                    padding-left:10px;
                    color:white;
                '>
                    Optimization Running, please be patient...
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
        # This is where we first start instantiating actual widgets.

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
        # AES

        self.save_opt_history_button = self._init_html_download_button(
            title="Save Optimization History",
            mimetype="text/plain",
        )

        # Optimizer widgets
        # Just use same styling as top-level drive widgets (not accordion)
        opt_dropdown_style = {"description_width": "120px"}

        self.widget_opt_obj_fun = Dropdown(
            options=["dipole_rmse", "maximize_psd"],
            value="dipole_rmse",
            description="Objective Function:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_solver = Dropdown(
            options=["bayesian", "cobyla"],
            value="bayesian",
            description="Solver:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_max_iter = BoundedIntText(
            value=5,
            min=1,
            max=10000,
            description="Max Iterations:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_tstop = BoundedFloatText(
            value=170,
            min=1,
            max=1000.0,
            description="tstop (ms):",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_n_jobs = BoundedIntText(
            value=1,
            min=1,
            max=self.n_cores,
            description="Cores:",
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_dt = BoundedFloatText(
            value=0.025,
            description="dt (ms):",
            min=0,
            max=10,
            step=0.01,
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_smoothing = BoundedFloatText(
            value=30.0,
            description="Dipole Smoothing:",
            min=0.0,
            max=100.0,
            step=1.0,
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )
        self.widget_opt_scaling = FloatText(
            value=3000.0,
            description="Dipole Scaling:",
            step=100.0,
            disabled=False,
            layout=self.layout["opt_textbox"],
            style=opt_dropdown_style,
        )

        self.run_opt_button = create_expanded_button(
            "Run Optimization",
            "success",
            layout=Layout(
                height=self.layout["run_btn"].height,
                width="74%",
            ),
            button_color=self.layout["theme_color"],
        )

        # end of AES
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

        # AES
        # Add optimization section
        self.opt_drive_widgets = list()
        self.opt_drive_boxes = list()
        self.opt_target_widgets = {}
        self.opt_drive_accordion = Accordion()
        self.opt_results = list()

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
        self._drives_out = Output().add_class("external-drives-accordion-widgets")
        self._connectivity_out = Output().add_class("connectivity-accordion-widgets")
        self._cell_params_out = Output().add_class("cell-parameters-widgets")
        self._global_gain_out = Output().add_class("connectivity-gains-widgets")
        # AES
        self._opt_target_out = Output()  # dynamic target params widgets of opt tab
        self._opt_drives_out = Output()  # drive accordion of optimization tab
        # end of AES

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

        # generate the HTML contents for the title-bar
        # --------------------------------------------------
        # The sun and moon icons for the theme toggle button are kept here so that
        # they load immediately with the GUI. we *could* move these to the .js, but
        # then there would be a delay before the buttons appears.
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
                            onclick="hnnToggleTheme()">
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
            output = self.add_drive_tab_drive_widget(
                self.widget_drive_type_selection.value,
                location,
            )
            self.add_opt_tab_drive_widget(
                drive_type=self.widget_drive_type_selection.value,
                location=location,
                prespecified_drive_name=self.drive_widgets[-1]["name"],
                drive_idx=-1,
            )
            return output

        def _delete_drives_clicked(b):
            self._drives_out.clear_output()
            self._opt_drives_out.clear_output()
            # black magic: the following does not work
            # global drive_widgets; drive_widgets = list()
            while len(self.drive_widgets) > 0:
                self.drive_widgets.pop()
            while len(self.drive_boxes) > 0:
                self.drive_boxes.pop()
            while len(self.opt_drive_widgets) > 0:
                self.opt_drive_widgets.pop()
            while len(self.opt_drive_boxes) > 0:
                self.opt_drive_boxes.pop()

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
            result = run_opt_button_clicked(
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
                self.widget_opt_smoothing.value,
                self.widget_opt_scaling.value,
                self.opt_target_widgets,
            )
            # AES "Hack" to re-load our NEW, optimized drive parameters...
            # ...is it a hack if it works well? ;)
            if result:
                output_config, opt_result = result
                # Rebuild the drives' widgets in the Drives and Optimization tabs from
                # the returned parameters after optimization:
                self.params = json.loads(output_config)
                self.load_conn_drives_opt_widgets()

                # Add and use the output of the latest optimization run to the history:
                self.opt_results.append(opt_result)
                self._update_opt_history_button()
            return

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

        def _opt_obj_fun_change(value):
            self._update_opt_target_hbox(value.new)

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

        # Many Optimization tab observations, including dual-linking widgets with their
        # equivalent in the Run tab:
        self.widget_opt_obj_fun.observe(_opt_obj_fun_change, "value")

        link(
            (self.widget_opt_tstop, "value"),
            (self.widget_tstop, "value"),
        )
        link(
            (self.widget_opt_dt, "value"),
            (self.widget_dt, "value"),
        )
        link(
            (self.widget_opt_n_jobs, "value"),
            (self.widget_n_jobs, "value"),
        )
        link(
            (self.widget_default_smoothing, "value"),
            (self.widget_opt_smoothing, "value"),
        )
        link(
            (self.widget_default_scaling, "value"),
            (self.widget_opt_scaling, "value"),
        )

    def _delete_single_drive(self, b):
        index = self.drive_accordion.selected_index

        # Remove selected drive from drive lists
        self.drive_boxes.pop(index)
        self.drive_widgets.pop(index)
        # Do the same for the Optimization tab's representation of that drive
        self.opt_drive_boxes.pop(index)
        self.opt_drive_widgets.pop(index)

        # Rebuild the accordion collections
        self.drive_accordion.titles = tuple(
            t for i, t in enumerate(self.drive_accordion.titles) if i != index
        )
        self.drive_accordion.selected_index = None
        self.drive_accordion.children = self.drive_boxes

        self.opt_drive_accordion.titles = tuple(
            t for i, t in enumerate(self.opt_drive_accordion.titles) if i != index
        )
        self.opt_drive_accordion.selected_index = None
        self.opt_drive_accordion.children = self.opt_drive_boxes

        # Render
        self._drives_out.clear_output()
        with self._drives_out:
            display(self.drive_accordion)
        self._opt_drives_out.clear_output()
        with self._opt_drives_out:
            display(self.opt_drive_accordion)

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
                # The dynamic-spacer can shrink/grow based on available space to
                # help prevent any "smushing" or overlap that may appear on some
                # OS/browser combinations but not others:
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
                # The VBox below contains the run, save, and load buttons, as well as
                # the dropdown widget for selecting networks/simulations to save:
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
                # The connectivity weights/gains accordion, already built from
                # _init_ui_components
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
                # the external drives accordion, already built from _init_ui_components
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

        # AES
        # Create optimization tab
        # -----------------------------------------
        # The Optimization tab is divided into 4 main sections:
        # 1. The always-shown top-level optimization parameters
        # 2. The "target" parameters box (dynamic, depending on your objective function)
        # 3. The always-shown "Run Optimization" and "Save Optimization History" buttons
        # 4. The drives accordion (dynamic, depending on existing drives in the Drives tab)
        opt_box = VBox(
            [
                # 1. Top-level optimization parameters
                HBox(
                    [
                        VBox(
                            [
                                self.widget_opt_obj_fun,
                                self.widget_opt_solver,
                                self.widget_opt_n_jobs,
                                self.widget_opt_smoothing,
                            ]
                        ),
                        VBox(
                            [
                                self.widget_opt_max_iter,
                                self.widget_opt_tstop,
                                self.widget_opt_dt,
                                self.widget_opt_scaling,
                            ]
                        ),
                    ]
                ),
                # 2. Target parameters box (a dynamic Output widget)
                self._opt_target_out,
                # 3. Run and Save History buttons
                HBox(
                    [
                        self.run_opt_button,
                        self.save_opt_history_button,
                    ],
                    layout=Layout(width="100%"),
                ),
                # 4. Drives accordion (a dynamic Output widget)
                self._opt_drives_out,
            ]
        )
        # end of AES

        # build the param_window_tabs_widget Tab() object, which holds both the
        # tab bar *and* the associated contents for each tab
        param_window_tabs_widget = Tab().add_class("param-window-tabs-widget")
        param_window_tabs_widget.layout = self.layout["param_window_tabs_widget"]

        # assign tab contents
        param_window_tabs_widget.children = [
            simulation_tab_contents,
            network_tab_contents,
            drive_tab_contents,
            opt_box,  # AES
            visualization_tab_contents,
        ]

        # set each tab's title
        titles = (
            "Simulation",
            "Network",
            "External drives",
            "Optimization",
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

    def load_custom_gui_styling(self):
        """
        Load custom CSS and JS for styling the GUI
        """
        gui_directory = Path(__file__).parent
        css_path = gui_directory / "gui_styles.css"
        js_path = gui_directory / "gui_scripts.js"

        with open(css_path, "r") as f:
            gui_styles = f.read()
        with open(js_path, "r") as f:
            gui_scripts = f.read()

        # ----------------------------------------------------------------------
        # JavaScript tags injected via HTML() (which is how IPywidgets updates
        # the DOM) are not executed by browsers for security reasons. To get
        # around this constraint, we use a 1x1 transparent GIF to trigger the "onload"
        # event that follows it. This method may seem a bit "hacky", but it is
        # actually a very commonly used trick for injecting scripts into IPywidgets.
        # Because "onload" is a "lifecycle event" of a valid asset (basically, a
        # high-priority attribute that must be processed when rendering the web page),
        # the browser is forced to execute the included JavaScript as soon as the
        # widget is rendered. This gives us a reliable execution hook that
        # is responsive to changes to the GUI (such as tabs being added or closed).
        # ----------------------------------------------------------------------

        # "escape" any double quotes in our JS code, which is needed since we insert
        # our script file as a string via the HTML attribute onload="{gui_scripts}"
        gui_scripts = gui_scripts.replace('"', "&quot;")

        minimal_img_src = (
            "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP"
            + "///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        )

        custom_assets = f"""
            <style>
                {gui_styles}
            </style>
            <img
                src="{minimal_img_src}"
                onload="{gui_scripts}"
                style="display:none;"
            >
        """

        return custom_assets

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
        custom_gui_styling = self.load_custom_gui_styling()
        self._header.value = custom_gui_styling + self._header.value

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
        # Note: see __init__() for diagrams and explanations of the overall structure,
        # including nesting and HTML tag usage.
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

        # initialize connectivity, drives, and optimization ipywidgets
        self.load_conn_drives_opt_widgets()

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

    def load_conn_drives_opt_widgets(self):
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
            self.update_drive_tab_accordion(self.params)

            # Add optimization
            self.update_opt_tab_target_widgets()
            self.update_opt_tab_accordion(self.params)

    def add_drive_tab_drive_widget(
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
        if drive_type == "Tonic" and not _check_if_tonic_bias_exists(
            self.drive_widgets
        ):
            return

        # Build drive widget objects
        name = (
            drive_type + str(len(self.drive_boxes))
            if not prespecified_drive_name
            else prespecified_drive_name
        )
        drive_tab_var_layout = self.layout["drive_textbox"]
        drive_tab_var_style = {"description_width": "125px"}

        prespecified_drive_data = (
            {} if not prespecified_drive_data else prespecified_drive_data
        )
        prespecified_drive_data.update({"seedcore": max(event_seed, 2)})

        choose_tab_drive_or_opt = "drive"
        new_drive_widgets, new_drive_box = _create_widgets_for_drive(
            drive_type,
            name,
            self.widget_tstop,
            location,
            choose_tab_drive_or_opt,
            prespecified_drive_data,
            prespecified_weights_ampa,
            prespecified_weights_nmda,
            prespecified_delays,
            prespecified_n_drive_cells,
            prespecified_cell_specific,
            drive_tab_var_layout=drive_tab_var_layout,
            drive_tab_var_style=drive_tab_var_style,
        )

        # Add delete button and assign its call-back function
        delete_button = Button(
            description="Delete",
            button_style="danger",
            icon="close",
            layout=self.layout["del_fig_btn"],
        ).add_class("red-button")
        delete_button.on_click(self._delete_single_drive)
        new_drive_box.children += (
            HTML(value="<p> </p>"),  # Adds blank space
            delete_button,
        )

        self.drive_boxes.append(new_drive_box)
        self.drive_widgets.append(new_drive_widgets)

        if render:
            # Construct accordion object
            self.drive_accordion.children = self.drive_boxes
            self.drive_accordion.selected_index = (
                len(self.drive_boxes) - 1 if expand_last_drive else None
            )
            # Update accordion title with location
            for idx, new_drive_widgets in enumerate(self.drive_widgets):
                tab_name = new_drive_widgets["name"]
                if new_drive_widgets["type"] != "Tonic":
                    tab_name += f" ({new_drive_widgets['location']})"
                self.drive_accordion.set_title(idx, tab_name)

            self._drives_out.clear_output()
            with self._drives_out:
                display(self.drive_accordion)

    def update_drive_tab_accordion(self, params):
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
            self.add_drive_tab_drive_widget(
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
                self.update_drive_tab_accordion(params)
                self.update_opt_tab_target_widgets()
                self.update_opt_tab_accordion(params)
            else:
                raise ValueError

            print(f"Loaded {load_type} from {param_dict['name']}")
        # Resets file counter to 0
        change["owner"].set_trait("value", ([]))
        return params

    def _update_opt_target_hbox(self, opt_obj_fun):
        self._opt_target_out.clear_output()

        if opt_obj_fun == "dipole_rmse":
            displayed_target_widgets = HBox(
                [
                    self.opt_target_widgets["rmse_target_data"],
                    self.opt_target_widgets["n_trials"],
                ]
            )
        elif opt_obj_fun == "maximize_psd":
            displayed_target_widgets = VBox(
                [
                    HBox(
                        [
                            HTML("Frequency Band 1 (Hz)"),
                            HTML(
                                "<span style='display:inline-block;width: 32px;'></span>"
                            ),
                            self.opt_target_widgets["psd_target_band1_min"],
                            self.opt_target_widgets["psd_target_band1_max"],
                            self.opt_target_widgets["psd_target_band1_proportion"],
                        ]
                    ),
                    HBox(
                        [
                            HTML("Frequency Band 2 (Hz)"),
                            self.opt_target_widgets["psd_target_band2_checkbox"],
                            self.opt_target_widgets["psd_target_band2_min"],
                            self.opt_target_widgets["psd_target_band2_max"],
                            self.opt_target_widgets["psd_target_band2_proportion"],
                        ]
                    ),
                ]
            )

        with self._opt_target_out:
            display(displayed_target_widgets)

    def update_opt_tab_target_widgets(self):
        # Preserve prior widget state for "target data" if widgets already exist
        # ------------------------------------------------------------------------------
        prior_target_state = {}
        # Target data widgets are unfortunately reset after run. The below functions to restore the
        # previous states:
        if self.opt_target_widgets:
            # Save current state of all target widgets
            for key, widget in self.opt_target_widgets.items():
                if hasattr(widget, "value"):
                    prior_target_state[key] = widget.value

        # The obj_fun="dipole_rmse" case is very simple
        # ------------------------------------------------------------------------------
        self.opt_target_widgets["rmse_target_data"] = Dropdown(
            options=self.data["simulation_data"].keys(),
            value=prior_target_state.get("rmse_target_data", None),
            description="Target Data:",
            disabled=False,
            layout=Layout(width="500px"),
            style={"description_width": "80px"},
        )
        # Set `_external_data_widget` to `opt_target_widgets["rmse_target_data"]` when simulation
        # data changes
        self.viz_manager._external_data_widget = self.opt_target_widgets[
            "rmse_target_data"
        ]

        self.opt_target_widgets["n_trials"] = IntText(
            value=prior_target_state.get("n_trials", 1),
            description="Trials:",
            disabled=False,
            layout=Layout(width="120px"),
            style={"description_width": "60px"},
        )
        link(
            (self.opt_target_widgets["n_trials"], "value"),
            (self.widget_ntrials, "value"),
        )

        # The obj_fun="maximize_psd" case is much more complex
        # ------------------------------------------------------------------------------
        # Note: these are ONLY for the Optimization "target" widgets, NOT the
        # Optimization drive widgets!
        #
        # Visual config for checkbox widgets
        checkbox_layout = Layout(width="30px")
        checkbox_style = {"description_width": "0px"}
        # Visual config for min and max frequency band widgets
        minmax_layout = Layout(width="90px")
        minmax_style = {"description_width": "30px"}
        # Visual config for frequency band proportion widgets
        proportion_layout = Layout(width="140px")
        proportion_style = {"description_width": "80px"}

        # Parameters for obj_fun="maximize_psd" Frequency Band 1
        # ------------------------------------------------------------------------------
        # Determine disabled states based on prior checkbox value (if it exists)
        band2_checkbox_value = prior_target_state.get(
            "psd_target_band2_checkbox", False
        )
        band1_proportion_disabled = not band2_checkbox_value
        band2_widgets_disabled = not band2_checkbox_value

        self.opt_target_widgets["psd_target_band1_min"] = BoundedFloatText(
            value=prior_target_state.get("psd_target_band1_min", 15),
            description="Min:",
            min=0,
            max=1e6,
            step=0.1,
            layout=minmax_layout,
            style=minmax_style,
        )
        self.opt_target_widgets["psd_target_band1_max"] = BoundedFloatText(
            value=prior_target_state.get("psd_target_band1_max", 25),
            description="Max:",
            min=0,
            max=1e6,
            step=0.1,
            layout=minmax_layout,
            style=minmax_style,
        )
        self.opt_target_widgets["psd_target_band1_proportion"] = BoundedFloatText(
            value=prior_target_state.get("psd_target_band1_proportion", 1),
            description="Proportion:",
            min=0,
            max=1,
            step=0.1,
            disabled=band1_proportion_disabled,
            layout=proportion_layout,
            style=proportion_style,
        )

        # Parameters for obj_fun="maximize_psd" Frequency Band 2
        # ------------------------------------------------------------------------------
        self.opt_target_widgets["psd_target_band2_checkbox"] = Checkbox(
            value=band2_checkbox_value,
            layout=checkbox_layout,
            style=checkbox_style,
        )
        self.opt_target_widgets["psd_target_band2_min"] = BoundedFloatText(
            value=prior_target_state.get("psd_target_band2_min", 9),
            description="Min:",
            min=0,
            max=1e6,
            step=0.1,
            disabled=band2_widgets_disabled,
            layout=minmax_layout,
            style=minmax_style,
        )
        self.opt_target_widgets["psd_target_band2_max"] = BoundedFloatText(
            value=prior_target_state.get("psd_target_band2_max", 14),
            description="Max:",
            min=0,
            max=1e6,
            step=0.1,
            disabled=band2_widgets_disabled,
            layout=minmax_layout,
            style=minmax_style,
        )
        self.opt_target_widgets["psd_target_band2_proportion"] = BoundedFloatText(
            value=prior_target_state.get("psd_target_band2_proportion", 0),
            description="Proportion:",
            min=0,
            max=1,
            step=0.1,
            disabled=band2_widgets_disabled,
            layout=proportion_layout,
            style=proportion_style,
        )

        # Let's have the PSD band2 checkbox control the ghosting/disabling of its other
        # band2 widgets, AND the band1 proportion widget
        # ------------------------------------------------------------------------------
        def _band2_ghosting_callback(change):
            if self.opt_target_widgets["psd_target_band2_min"].disabled:
                self.opt_target_widgets["psd_target_band1_proportion"].disabled = False
                self.opt_target_widgets["psd_target_band2_min"].disabled = False
                self.opt_target_widgets["psd_target_band2_max"].disabled = False
                self.opt_target_widgets["psd_target_band2_proportion"].disabled = False
            else:
                self.opt_target_widgets["psd_target_band1_proportion"].disabled = True
                # Set (or reset) the proportion of band1 to 1, since it's the only
                # active band
                self.opt_target_widgets["psd_target_band1_proportion"].value = 1
                self.opt_target_widgets["psd_target_band2_min"].disabled = True
                self.opt_target_widgets["psd_target_band2_max"].disabled = True
                self.opt_target_widgets["psd_target_band2_proportion"].disabled = True
                # Set (or reset) the proportion of band2 to 0, since it's not active
                self.opt_target_widgets["psd_target_band2_proportion"].value = 0

        self.opt_target_widgets["psd_target_band2_checkbox"].observe(
            _band2_ghosting_callback,
            names="value",
        )

        # Next, let's make the two Proportion widgets inter-dependent and add to 1
        # ------------------------------------------------------------------------------
        # Flag to prevent infinite recursion
        _updating = {"flag": False}

        def update_widget1(change):
            if _updating["flag"]:
                return
            _updating["flag"] = True
            self.opt_target_widgets["psd_target_band1_proportion"].value = (
                1.0 - change["new"]
            )
            _updating["flag"] = False

        def update_widget2(change):
            if _updating["flag"]:
                return
            _updating["flag"] = True
            self.opt_target_widgets["psd_target_band2_proportion"].value = (
                1.0 - change["new"]
            )
            _updating["flag"] = False

        self.opt_target_widgets["psd_target_band1_proportion"].observe(
            update_widget2, names="value"
        )
        self.opt_target_widgets["psd_target_band2_proportion"].observe(
            update_widget1, names="value"
        )

        # FINALLY, actually display all this stuff
        # ------------------------------------------------------------------------------
        self._update_opt_target_hbox(
            self.widget_opt_obj_fun.value,
        )

    def _update_opt_history_button(self):
        """Update the optimization history download button with current data."""
        # Create the timestamps
        time_now = datetime.now()
        report_timestamp = time_now.strftime("%Y-%m-%d %H:%M:%S")
        filename_timestamp = time_now.strftime("%Y%m%d_%H%M%S")

        # Generate the table content
        table_content = generate_opt_history_table(self.opt_results, report_timestamp)

        # Encode the content for download
        b64 = base64.b64encode(table_content.encode())
        payload = b64.decode()

        filename = f"optimization_history_{filename_timestamp}.txt"

        # Update the button
        self.save_opt_history_button.value = self.html_download_button.format(
            payload=payload,
            filename=filename,
            is_disabled="",
            btn_height=self.layout["run_btn"].height,
            color_theme=self.layout["theme_color"],
            title="Save Optimization History",
            mimetype="text/plain",
        )

    def update_opt_tab_accordion(self, params):
        """Create/update the drives output of the optimization tab"""
        net = dict_to_network(params)
        drive_specs = net.external_drives
        tonic_specs = net.external_biases

        prior_opt_widget_values = {}
        # This is a check for if there's existing "state" in the Optimization drive widgets:
        if self.opt_drive_widgets:
            # This means there's existing "state" in the Optimization drive widgets, such as if
            # we're doing a second round of optimization after the user has indicated that they want
            # to optimize against a particular parameter (such as by checking its checkbox). In this
            # case, we want to re-load the prior "state" of our optimization widgets:
            #
            # This code is basically copied from `_generate_constraints_and_func`, but uses
            # `_build_constraints` to find any prior min/max percentages for the widget values,
            # instead of the actual "true" value of the constrained parameter.
            for drive in self.opt_drive_widgets:
                if drive["type"] in ("Tonic"):
                    prior_opt_widget_values.update(_build_constraints(drive))
                    prior_opt_widget_values.update(
                        _build_constraints(drive, syn_type="amplitude")
                    )
                else:
                    # Synaptic variables are a special case, since they are dicts instead of
                    # single values
                    for syn_type in ("weights_ampa", "weights_nmda", "delays"):
                        prior_opt_widget_values.update(
                            _build_constraints(drive, syn_type=syn_type)
                        )
                    if drive["type"] == "Poisson":
                        prior_opt_widget_values.update(_build_constraints(drive))
                        prior_opt_widget_values.update(
                            _build_constraints(drive, syn_type="rate_constant")
                        )
                    elif drive["type"] in ("Evoked", "Gaussian"):
                        prior_opt_widget_values.update(_build_constraints(drive))
                    elif drive["type"] in ("Rhythmic", "Bursty"):
                        prior_opt_widget_values.update(_build_constraints(drive))

        # clear before adding drives
        self._opt_drives_out.clear_output()
        while len(self.opt_drive_widgets) > 0:
            self.opt_drive_widgets.pop()
            self.opt_drive_boxes.pop()

        drive_names = list(drive_specs.keys())
        # Add tonic biases
        if tonic_specs:
            drive_names.extend(list(tonic_specs.keys()))

        for drive_idx, drive_name in enumerate(drive_names):  # order matters
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

            should_render = drive_idx == (len(drive_names) - 1)

            self.add_opt_tab_drive_widget(
                drive_type=specs["type"].capitalize(),
                location=specs["location"],
                prespecified_drive_name=drive_name,
                render=should_render,
                expand_last_drive=False,
                drive_idx=drive_idx,
                prior_opt_widget_values=prior_opt_widget_values,
                **kwargs,
            )

    def add_opt_tab_drive_widget(
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
        drive_idx=None,
        event_seed=14,
        prior_opt_widget_values=None,
    ):
        """Add a optimization widget for a new drive, including to the accordion."""

        # Check only adds 1 tonic input widget to the OPT per-drive widgets (the
        # self.drive_widgets, on the other hand, have already been built by this point,
        # and if there is a tonic drive, it will be present in self.drive_widgets but
        # not in self.opt_drive_widgets):
        if drive_type == "Tonic" and not _check_if_tonic_bias_exists(
            self.opt_drive_widgets
        ):
            return

        name = (
            drive_type + str(len(self.drive_boxes))
            if not prespecified_drive_name
            else prespecified_drive_name
        )
        prespecified_drive_data = (
            {} if not prespecified_drive_data else prespecified_drive_data
        )
        prespecified_drive_data.update({"seedcore": max(event_seed, 2)})

        # Set the lower (100% - X) and upper (100% + X) bounds of the initial min/max
        # constraint values. This should NEVER EXCEED 100!
        initial_constraint_range_percentage = 50

        # Stylin'
        # ------------------------------------------------------------------------------
        # Visual config for "main variable" widgets
        opt_tab_var_layout = Layout(width="230px")
        opt_tab_var_style = {"description_width": "120px"}
        # Visual config for checkbox widgets
        opt_tab_checkbox_layout = Layout(width="30px")
        opt_tab_checkbox_style = {"description_width": "0px"}
        # Visual config for min and max constraint widgets
        opt_tab_minmax_layout = Layout(width="100px")
        opt_tab_minmax_style = {"description_width": "30px"}
        opt_tab_quad_hbox_layout = Layout(
            display="flex",
            flex_flow="row",
            align_items="flex-start",
            width="480px",  # carefully curated...
        )
        html_tab = "&emsp;"
        opt_tab_column_titles = HTML(
            value=f"""
            <div style='margin:0px 0px 0px 190px;'><b>Optimize against?</b>
            {html_tab}{html_tab}{html_tab}Constraints (in %):</div>
            """,
        )

        choose_tab_drive_or_opt = "opt"
        opt_drive_widget, opt_drive_box = _create_widgets_for_drive(
            drive_type,
            name,
            self.widget_tstop,
            location,
            choose_tab_drive_or_opt,
            prespecified_drive_data,
            prespecified_weights_ampa,
            prespecified_weights_nmda,
            prespecified_delays,
            prespecified_n_drive_cells,
            prespecified_cell_specific,
            opt_tab_quad_hbox_layout=opt_tab_quad_hbox_layout,
            opt_tab_column_titles=opt_tab_column_titles,
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=self.drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )

        self.opt_drive_boxes.append(opt_drive_box)
        self.opt_drive_widgets.append(opt_drive_widget)

        if render:
            # Construct accordion object
            self.opt_drive_accordion.children = self.opt_drive_boxes
            self.opt_drive_accordion.selected_index = (
                len(self.opt_drive_boxes) - 1 if expand_last_drive else None
            )
            # Update accordion title with location
            for idx, opt_drive_widget in enumerate(self.opt_drive_widgets):
                tab_name = opt_drive_widget["name"]
                if opt_drive_widget["type"] != "Tonic":
                    tab_name += f" ({opt_drive_widget['location']})"
                self.opt_drive_accordion.set_title(idx, tab_name)

            self._opt_drives_out.clear_output()
            with self._opt_drives_out:
                display(self.opt_drive_accordion)


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


def _create_synaptic_widgets(
    location,
    choose_tab_drive_or_opt,  # which tab to build for, including which kwargs to use
    data={},
    # Drive-tab-specific kwargs
    drive_tab_var_layout=None,
    drive_tab_var_style=None,
    # Optimization-tab-specific kwargs
    if_poisson=False,
    opt_tab_quad_hbox_layout=None,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """Create synaptic weight, delay, and (optionally) rate constant widgets.

    When ``choose_tab_drive_or_opt=="drive"``, this creates plain widgets for the Drives
    tab. When ``choose_tab_drive_or_opt=="opt"``, this creates Optimization widgets with
    constraint controls and observers that mirror the Drives-tab widgets.

    This handles the cases for Evoked, Poisson, and Rhythmic drives, but not
    Tonic biases.
    """
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
    if if_poisson:
        default_data.update(
            {
                "rate_constant": {
                    "L2_pyramidal": 140.0,
                    "L5_pyramidal": 40.0,
                    "L5_basket": 40.0,
                    "L2_basket": 40.0,
                },
            }
        )

    data = _update_nested_dict(default_data, data)

    # Setup our styling
    if choose_tab_drive_or_opt == "opt":
        simple_widget_kwargs = dict(layout=opt_tab_var_layout, style=opt_tab_var_style)
        complex_opt_widget_kwargs = dict(
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )
    elif choose_tab_drive_or_opt == "drive":
        simple_widget_kwargs = dict(
            layout=drive_tab_var_layout, style=drive_tab_var_style
        )
        # No complex widget kwargs needed for non-Optimization widgets

    cell_types = ["L5_pyramidal", "L2_pyramidal", "L5_basket", "L2_basket"]
    if location == "distal":
        cell_types.remove("L5_basket")

    # Initialize the key output container, which will contain all the widgets
    syn_widgets_dict = {
        "weights_ampa": {},
        "weights_nmda": {},
        "delays": {},
    }
    if if_poisson:
        syn_widgets_dict["rate_constant"] = {}

    if choose_tab_drive_or_opt == "opt":
        opt_widgets_box_list = {
            "weights_ampa": [],
            "weights_nmda": [],
            "delays": [],
        }
        if if_poisson:
            opt_widgets_box_list["rate_constant"] = []

        for cell_type in cell_types:
            for syn_param_type in syn_widgets_dict.keys():
                syn_widgets_dict[syn_param_type].update(
                    _create_opt_widgets_for_drive_var(
                        cell_type,
                        data[syn_param_type][cell_type],
                        f"{cell_type}:",
                        syn_type=syn_param_type,
                        **complex_opt_widget_kwargs,
                    )
                )
                opt_widgets_box_list[syn_param_type].append(
                    _create_hbox_for_opt_var(
                        cell_type,
                        syn_widgets_dict[syn_param_type],
                        opt_tab_quad_hbox_layout,
                    )
                )

        syn_widgets_list = (
            [HTML(value="<b>AMPA weights</b>")]
            + opt_widgets_box_list["weights_ampa"]
            + [HTML(value="<b>NMDA weights</b>")]
            + opt_widgets_box_list["weights_nmda"]
            + [HTML(value="<b>Synaptic delays</b>")]
            + opt_widgets_box_list["delays"]
            + (
                (
                    [HTML(value="<b>Rate constants</b>")]
                    + opt_widgets_box_list["rate_constant"]
                )
                if if_poisson
                else []
            )
        )
    elif choose_tab_drive_or_opt == "drive":
        for cell_type in cell_types:
            for syn_param_type in syn_widgets_dict.keys():
                syn_widgets_dict[syn_param_type].update(
                    {
                        f"{cell_type}": BoundedFloatText(
                            value=data[syn_param_type][cell_type],
                            description=f"{cell_type}:",
                            min=0,
                            max=1e6,
                            step=0.01,
                            **simple_widget_kwargs,
                        )
                    }
                )

        syn_widgets_list = (
            [HTML(value="<b>AMPA weights</b>")]
            + list(syn_widgets_dict["weights_ampa"].values())
            + [HTML(value="<b>NMDA weights</b>")]
            + list(syn_widgets_dict["weights_nmda"].values())
            + [HTML(value="<b>Synaptic delays</b>")]
            + list(syn_widgets_dict["delays"].values())
            + (
                (
                    [HTML(value="<b>Rate constants</b>")]
                    + list(syn_widgets_dict["rate_constant"].values())
                )
                if if_poisson
                else []
            )
        )

    return syn_widgets_list, syn_widgets_dict


def _cell_spec_change(change, widget):
    if change["new"]:
        widget.disabled = True
    else:
        widget.disabled = False


def _create_widgets_for_evoked(
    name,
    location,
    choose_tab_drive_or_opt,  # which tab to build for, including which kwargs to use
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
    # Drive-tab-specific kwargs
    drive_tab_var_layout=None,
    drive_tab_var_style=None,
    # Optimization-tab-specific kwargs
    opt_tab_quad_hbox_layout=None,
    opt_tab_column_titles=None,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """Create all widgets (& observers) for an Evoked drive.

    When ``choose_tab_drive_or_opt=="drive"``, this creates the widgets for the Drives
    tab. When ``choose_tab_drive_or_opt=="opt"``, this creates the widgets for the
    Optimization tab, including constraint widgets and observers that mirror the
    Drives-tab widgets.
    """
    # Initialize our data dict with default values, then overwrite with any passed
    # values:
    default_data = {
        "mu": 0,
        "sigma": 1,
        "numspikes": 1,
        "n_drive_cells": 1,
        "cell_specific": True,
        "seedcore": 14,
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    data = _update_nested_dict(default_data, data)

    # Set our layout and styling preferences for the widgets according to which tab
    # we're building for:
    if choose_tab_drive_or_opt == "opt":
        simple_widget_kwargs = dict(layout=opt_tab_var_layout, style=opt_tab_var_style)
        # Note that opt_tab_quad_hbox_layout and opt_tab_column_titles are not needed
        # here.
        complex_opt_widget_kwargs = dict(
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )
        syn_widget_kwargs = complex_opt_widget_kwargs
    elif choose_tab_drive_or_opt == "drive":
        simple_widget_kwargs = dict(
            layout=drive_tab_var_layout, style=drive_tab_var_style
        )
        # No complex widget kwargs needed for non-Optimization widgets
        syn_widget_kwargs = dict(
            drive_tab_var_layout=drive_tab_var_layout,
            drive_tab_var_style=drive_tab_var_style,
        )

    # Initialize the drive widget dict
    new_drive_widgets = dict(
        type="Evoked",
        name=name,
        location=location,
        sync_within_trial=False,
    )

    # mu and sigma widgets (including extra Optimization widgets)
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        new_drive_widgets.update(
            _create_opt_widgets_for_drive_var(
                "mu",
                data["mu"],
                "Mean time (ms):",
                **complex_opt_widget_kwargs,
            )
            | _create_opt_widgets_for_drive_var(
                "sigma",
                data["sigma"],
                "Std dev time (ms):",
                **complex_opt_widget_kwargs,
            )
        )
    elif choose_tab_drive_or_opt == "drive":
        mu = BoundedFloatText(
            value=data["mu"],
            description="Mean time:",
            min=0,
            max=1e6,
            step=0.01,
            **simple_widget_kwargs,
        )
        sigma = BoundedFloatText(
            value=data["sigma"],
            description="Std dev time:",
            min=0,
            max=1e6,
            step=0.01,
            **simple_widget_kwargs,
        )
        new_drive_widgets.update(dict(mu=mu, sigma=sigma))

    # Non-optimized widgets: numspikes, n_drive_cells, cell_specific, seedcore
    # --------------------------------------------------------------------------
    # Numspikes is a special case, since it MUST be an integer, but our
    # Optimization's constraints-updating functions currently assume all constraints are
    # floats, since they update according to fractional values. Therefore, we cannot
    # currently pass it to our constraints to use in Optimization currently.
    numspikes = BoundedIntText(
        value=data["numspikes"],
        description="No. Spikes:",
        min=0,
        max=int(1e6),
        **simple_widget_kwargs,
    )
    n_drive_cells = IntText(
        value=data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=data["cell_specific"],
        **simple_widget_kwargs,
    )
    cell_specific = Checkbox(
        value=data["cell_specific"],
        description="Cell-Specific",
        **simple_widget_kwargs,
    )
    seedcore = IntText(
        value=data["seedcore"], description="Seed:", **simple_widget_kwargs
    )

    # In the Optimization tab case, we want to cross-link these widgets with their
    # Simulation tab equivalents:
    if choose_tab_drive_or_opt == "opt":
        _make_opt_observers(numspikes, "numspikes", drive_widgets, drive_idx)
        _make_opt_observers(n_drive_cells, "n_drive_cells", drive_widgets, drive_idx)
        _make_opt_observers(cell_specific, "is_cell_specific", drive_widgets, drive_idx)
        _make_opt_observers(seedcore, "seedcore", drive_widgets, drive_idx)

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    # Update our outgoing collection of widgets:
    new_drive_widgets.update(
        dict(
            numspikes=numspikes,
            n_drive_cells=n_drive_cells,
            is_cell_specific=cell_specific,
            seedcore=seedcore,
        )
    )

    # Add the synaptic widgets. If creating them for the Optimization tab, we are
    # interested in constraining/optimizing against all synaptic parameters, and
    # therefore need to create widgets for all:
    # --------------------------------------------------------------------------
    syn_widgets_list, syn_widgets_dict = _create_synaptic_widgets(
        location,
        choose_tab_drive_or_opt,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
        opt_tab_quad_hbox_layout=opt_tab_quad_hbox_layout,
        **syn_widget_kwargs,
    )
    new_drive_widgets.update(syn_widgets_dict)

    # Finally, decide the "VBox" positioning of all of the above widgets
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        new_drive_box = VBox(
            [
                opt_tab_column_titles,
                _create_hbox_for_opt_var(
                    "mu", new_drive_widgets, opt_tab_quad_hbox_layout
                ),
                _create_hbox_for_opt_var(
                    "sigma", new_drive_widgets, opt_tab_quad_hbox_layout
                ),
                numspikes,
                n_drive_cells,
                cell_specific,
                seedcore,
            ]
            + syn_widgets_list
        )
    elif choose_tab_drive_or_opt == "drive":
        new_drive_box = VBox(
            [
                mu,
                sigma,
                numspikes,
                n_drive_cells,
                cell_specific,
                seedcore,
            ]
            + syn_widgets_list
        )

    return new_drive_widgets, new_drive_box


def _create_widgets_for_poisson(
    name,
    tstop_widget,
    location,
    choose_tab_drive_or_opt,  # which tab to build for, including which kwargs to use
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
    # Drive-tab-specific kwargs
    drive_tab_var_layout=None,
    drive_tab_var_style=None,
    # Optimization-tab-specific kwargs
    opt_tab_quad_hbox_layout=None,
    opt_tab_column_titles=None,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """Create all widgets (& observers) for a Poisson drive.

    When ``choose_tab_drive_or_opt=="drive"``, this creates the widgets for the Drives
    tab. When ``choose_tab_drive_or_opt=="opt"``, this creates the widgets for the
    Optimization tab, including constraint widgets and observers that mirror the
    Drives-tab widgets.
    """
    # Initialize our data dict with default values, then overwrite with any passed
    # values. The default rate constants are available in `_create_synaptic_widgets`.
    default_data = {
        "tstart": 0.0,
        "tstop": tstop_widget.value,
        "n_drive_cells": 1,
        "cell_specific": True,
        "seedcore": 14,
    }
    data.update({"n_drive_cells": n_drive_cells, "cell_specific": cell_specific})
    data = _update_nested_dict(default_data, data)

    # Set our layout and styling preferences for the widgets according to which tab
    # we're building for:
    if choose_tab_drive_or_opt == "opt":
        simple_widget_kwargs = dict(layout=opt_tab_var_layout, style=opt_tab_var_style)
        # Note that opt_tab_quad_hbox_layout and opt_tab_column_titles are not needed
        # here.
        complex_opt_widget_kwargs = dict(
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )
        syn_widget_kwargs = complex_opt_widget_kwargs
    elif choose_tab_drive_or_opt == "drive":
        simple_widget_kwargs = dict(
            layout=drive_tab_var_layout, style=drive_tab_var_style
        )
        # No complex widget kwargs needed for non-Optimization widgets
        syn_widget_kwargs = dict(
            drive_tab_var_layout=drive_tab_var_layout,
            drive_tab_var_style=drive_tab_var_style,
        )

    # Initialize the drive widget dict
    new_drive_widgets = dict(
        type="Poisson",
        name=name,
        location=location,  # notice this is not a widget but a str!
    )

    # Non-optimized widgets: tstart, tstop, n_drive_cells, cell_specific, seedcore
    # --------------------------------------------------------------------------
    tstart = BoundedFloatText(
        value=data["tstart"],
        description="Start time (ms)",
        min=0,
        max=1e6,
        **simple_widget_kwargs,
    )
    tstop = BoundedFloatText(
        value=data["tstop"],
        max=tstop_widget.value,
        description="Stop time (ms)",
        **simple_widget_kwargs,
    )
    n_drive_cells = IntText(
        value=data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=data["cell_specific"],
        **simple_widget_kwargs,
    )
    cell_specific = Checkbox(
        value=data["cell_specific"],
        description="Cell-Specific",
        **simple_widget_kwargs,
    )
    seedcore = IntText(
        value=data["seedcore"], description="Seed:", **simple_widget_kwargs
    )

    # In the Optimization tab case, we want to cross-link these widgets with their
    # Simulation tab equivalents:
    if choose_tab_drive_or_opt == "opt":
        _make_opt_observers(tstart, "tstart", drive_widgets, drive_idx)
        _make_opt_observers(tstop, "tstop", drive_widgets, drive_idx)
        _make_opt_observers(n_drive_cells, "n_drive_cells", drive_widgets, drive_idx)
        _make_opt_observers(cell_specific, "is_cell_specific", drive_widgets, drive_idx)
        _make_opt_observers(seedcore, "seedcore", drive_widgets, drive_idx)

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    # Update our outgoing collection of widgets:
    new_drive_widgets.update(
        dict(
            tstart=tstart,
            tstop=tstop,
            n_drive_cells=n_drive_cells,
            is_cell_specific=cell_specific,
            seedcore=seedcore,
        )
    )

    # Add the synaptic widgets. If creating them for the Optimization tab, we are
    # interested in constraining/optimizing against all synaptic parameters, and
    # therefore need to create widgets for all:
    # --------------------------------------------------------------------------
    syn_data = {
        "weights_ampa": weights_ampa,
        "weights_nmda": weights_nmda,
        "delays": delays,
    }
    # Since `rate_constant` is a Poisson-specific parameter, and its data is not passed
    # as an explicit argument to this _create_widgets_for_poisson, we extract its data
    # from `default_data` only if it's present. If it's not, then it will be filled in
    # by the default values inside _create_synaptic_widgets.
    #
    # Similarly, because "rate_constant" is extracted the same way that other synaptic
    # parameters are inside `_init_network_from_widgets`, we can package it along with
    # the other synaptic parameters.
    if "rate_constant" in default_data.keys():
        syn_data["rate_constant"] = default_data["rate_constant"]
    syn_widgets_list, syn_widgets_dict = _create_synaptic_widgets(
        location,
        choose_tab_drive_or_opt,
        data=syn_data,
        opt_tab_quad_hbox_layout=opt_tab_quad_hbox_layout,
        if_poisson=True,
        **syn_widget_kwargs,
    )
    new_drive_widgets.update(syn_widgets_dict)

    # Finally, decide the "VBox" positioning of all of the above widgets
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        new_drive_box = VBox(
            [
                opt_tab_column_titles,
                tstart,
                tstop,
                n_drive_cells,
                cell_specific,
                seedcore,
            ]
            + syn_widgets_list
        )
    elif choose_tab_drive_or_opt == "drive":
        new_drive_box = VBox(
            [tstart, tstop, n_drive_cells, cell_specific, seedcore] + syn_widgets_list
        )

    return new_drive_widgets, new_drive_box


def _create_widgets_for_rhythmic(
    name,
    tstop_widget,
    location,
    choose_tab_drive_or_opt,  # which tab to build for, including which kwargs to use
    data={},
    weights_ampa=None,
    weights_nmda=None,
    delays=None,
    n_drive_cells=None,
    cell_specific=None,
    # Drive-tab-specific kwargs
    drive_tab_var_layout=None,
    drive_tab_var_style=None,
    # Optimization-tab-specific kwargs
    opt_tab_quad_hbox_layout=None,
    opt_tab_column_titles=None,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """Create all widgets (& observers) for a Rhythmic drive.

    When ``choose_tab_drive_or_opt=="drive"``, this creates the widgets for
    the Drives tab. When ``choose_tab_drive_or_opt=="opt"``, this creates the widgets
    for the Optimization tab, including constraint widgets and observers that mirror the
    Drives-tab widgets.
    """
    # Initialize our data dict with default values, then overwrite with any passed
    # values:
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
    data = _update_nested_dict(default_data, data)

    # Set our layout and styling preferences for the widgets according to which tab
    # we're building for:
    if choose_tab_drive_or_opt == "opt":
        simple_widget_kwargs = dict(layout=opt_tab_var_layout, style=opt_tab_var_style)
        # Note that opt_tab_quad_hbox_layout and opt_tab_column_titles are not needed
        # here.
        complex_opt_widget_kwargs = dict(
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )
        syn_widget_kwargs = complex_opt_widget_kwargs
    elif choose_tab_drive_or_opt == "drive":
        simple_widget_kwargs = dict(
            layout=drive_tab_var_layout, style=drive_tab_var_style
        )
        # No complex widget kwargs needed for non-Optimization widgets
        syn_widget_kwargs = dict(
            drive_tab_var_layout=drive_tab_var_layout,
            drive_tab_var_style=drive_tab_var_style,
        )

    # Initialize the drive widget dict
    new_drive_widgets = dict(
        type="Rhythmic",
        name=name,
        location=location,
    )

    # burst_rate and burst_std widgets (including extra Optimization widgets)
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        new_drive_widgets.update(
            _create_opt_widgets_for_drive_var(
                "burst_rate",
                data["burst_rate"],
                "Burst rate (Hz)",
                **complex_opt_widget_kwargs,
            )
            | _create_opt_widgets_for_drive_var(
                "burst_std",
                data["burst_std"],
                "Burst std dev (Hz)",
                **complex_opt_widget_kwargs,
            )
        )
    elif choose_tab_drive_or_opt == "drive":
        burst_rate = BoundedFloatText(
            value=data["burst_rate"],
            description="Burst rate (Hz)",
            min=0,
            max=1e6,
            **simple_widget_kwargs,
        )
        burst_std = BoundedFloatText(
            value=data["burst_std"],
            description="Burst std dev (Hz)",
            min=0,
            max=1e6,
            **simple_widget_kwargs,
        )
        # Update our outgoing collection of widgets:
        new_drive_widgets.update(dict(burst_rate=burst_rate, burst_std=burst_std))

    # Non-optimized widgets: tstart, tstart_std, tstop, numspikes, n_drive_cells,
    # cell_specific, and seedcore
    # --------------------------------------------------------------------------
    tstart = BoundedFloatText(
        value=data["tstart"],
        description="Start time (ms)",
        min=0,
        max=1e6,
        **simple_widget_kwargs,
    )
    tstart_std = BoundedFloatText(
        value=data["tstart_std"],
        description="Start time dev (ms)",
        min=0,
        max=1e6,
        **simple_widget_kwargs,
    )
    tstop = BoundedFloatText(
        value=data["tstop"],
        description="Stop time (ms)",
        max=tstop_widget.value,
        **simple_widget_kwargs,
    )
    # Numspikes is a special case, since it MUST be an integer, but our Optimization's
    # constraints-updating functions currently assume all constraints are floats, since
    # they update according to fractional values. Therefore, we cannot currently pass it
    # to our constraints to use in Optimization currently.
    numspikes = BoundedIntText(
        value=data["numspikes"],
        description="No. Spikes:",
        min=0,
        max=int(1e6),
        **simple_widget_kwargs,
    )
    n_drive_cells = IntText(
        value=data["n_drive_cells"],
        description="No. Drive Cells:",
        disabled=data["cell_specific"],
        **simple_widget_kwargs,
    )
    cell_specific = Checkbox(
        value=data["cell_specific"],
        description="Cell-Specific",
        **simple_widget_kwargs,
    )
    seedcore = IntText(
        value=data["seedcore"], description="Seed:", **simple_widget_kwargs
    )

    # In the Optimization tab case, we want to cross-link these widgets with their
    # Simulation tab equivalents:
    if choose_tab_drive_or_opt == "opt":
        _make_opt_observers(tstart, "tstart", drive_widgets, drive_idx)
        _make_opt_observers(tstart_std, "tstart_std", drive_widgets, drive_idx)
        _make_opt_observers(tstop, "tstop", drive_widgets, drive_idx)
        _make_opt_observers(numspikes, "numspikes", drive_widgets, drive_idx)
        _make_opt_observers(n_drive_cells, "n_drive_cells", drive_widgets, drive_idx)
        _make_opt_observers(cell_specific, "is_cell_specific", drive_widgets, drive_idx)
        _make_opt_observers(seedcore, "seedcore", drive_widgets, drive_idx)

    # Disable n_drive_cells widget based on cell_specific checkbox
    cell_specific.observe(
        partial(_cell_spec_change, widget=n_drive_cells), names="value"
    )

    # Update our outgoing collection of widgets:
    new_drive_widgets.update(
        dict(
            tstart=tstart,
            tstart_std=tstart_std,
            tstop=tstop,
            numspikes=numspikes,
            n_drive_cells=n_drive_cells,
            is_cell_specific=cell_specific,
            seedcore=seedcore,
        )
    )

    # Add the synaptic widgets. If creating them for the Optimization tab, we are
    # interested in constraining/optimizing against all synaptic parameters, and
    # therefore need to create widgets for all:
    # --------------------------------------------------------------------------
    syn_widgets_list, syn_widgets_dict = _create_synaptic_widgets(
        location,
        choose_tab_drive_or_opt,
        data={
            "weights_ampa": weights_ampa,
            "weights_nmda": weights_nmda,
            "delays": delays,
        },
        opt_tab_quad_hbox_layout=opt_tab_quad_hbox_layout,
        **syn_widget_kwargs,
    )
    new_drive_widgets.update(syn_widgets_dict)

    # Finally, decide the "VBox" positioning of all of the above widgets
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        new_drive_box = VBox(
            [
                opt_tab_column_titles,
                tstart,
                tstart_std,
                tstop,
                _create_hbox_for_opt_var(
                    "burst_rate", new_drive_widgets, opt_tab_quad_hbox_layout
                ),
                _create_hbox_for_opt_var(
                    "burst_std", new_drive_widgets, opt_tab_quad_hbox_layout
                ),
                numspikes,
                n_drive_cells,
                cell_specific,
                seedcore,
            ]
            + syn_widgets_list
        )
    elif choose_tab_drive_or_opt == "drive":
        new_drive_box = VBox(
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
            + syn_widgets_list
        )

    return new_drive_widgets, new_drive_box


def _create_widgets_for_tonic(
    name,
    tstop_widget,
    choose_tab_drive_or_opt,  # which tab to build for, including which kwargs to use
    data=None,
    # Drive-tab-specific kwargs
    drive_tab_var_layout=None,
    drive_tab_var_style=None,
    # Optimization-tab-specific kwargs
    opt_tab_quad_hbox_layout=None,
    opt_tab_column_titles=None,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """Create all widgets (& observers) for a Tonic drive.

    When ``choose_tab_drive_or_opt=="drive"``, this creates the widgets for the Drives
    tab. When ``choose_tab_drive_or_opt=="opt"``, this creates the widgets for the
    Optimization tab, including constraint widgets and observers that mirror the
    Drives-tab widgets.
    """
    cell_types = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]

    # AES TODO: The original API for this function that creates the tonic bias widgets
    # had a small misconfiguration in its "data" argument. Specifically, it expected
    # "data" to consist of keys which are cell types and values which are dictionaries,
    # where the dictionaries include values for `t0`, `tstop`, and `amplitude`. However,
    # if you view the use of `add_tonic_bias` in `_init_network_from_widgets`, a
    # specific `amplitude` is used for each cell type, but the `t0` and `tstop` values
    # are shared across all cell types. Fixing this misconfiguration would require
    # changing how the "data" argument of this function is created and handled, and
    # refactoring of how the "Drives tab" case is handled (which is brought over from
    # the original code). Due to a current deadline, I will not be refactoring the tonic
    # Drive tab and Optimization code together.

    # Set our layout and styling preferences for the widgets
    # according to which tab we're building for:
    if choose_tab_drive_or_opt == "opt":
        simple_widget_kwargs = dict(layout=opt_tab_var_layout, style=opt_tab_var_style)
        # Note that opt_tab_quad_hbox_layout and opt_tab_column_titles are not needed
        # here.
        complex_opt_widget_kwargs = dict(
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )
    elif choose_tab_drive_or_opt == "drive":
        simple_widget_kwargs = dict(
            layout=drive_tab_var_layout, style=drive_tab_var_style
        )
        # No complex widget kwargs needed for non-Optimization widgets

    # Initialize our data dict with default values, then overwrite with any passed
    # values:
    if choose_tab_drive_or_opt == "opt":
        # Note that unlike the similar "weights_ampa" etc., the drive_widgets
        # use "amplitude" in the singular, not plural "amplitudes".
        default_data = {
            "t0": 0,
            "tstop": tstop_widget.value,
            "amplitude": {
                "L5_pyramidal": 0.0,
                "L2_pyramidal": 0.0,
                "L5_basket": 0.0,
                "L2_basket": 0.0,
            },
        }
    elif choose_tab_drive_or_opt == "drive":
        default_values = {"amplitude": 0, "t0": 0, "tstop": tstop_widget.value}
        t0 = default_values["t0"]
        tstop = default_values["tstop"]
        default_data = {cell_type: default_values for cell_type in cell_types}

    if isinstance(data, dict):
        data = _update_nested_dict(default_data, data)
    else:
        data = default_data

    # t0, tstop widgets
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        t0_widget = BoundedFloatText(
            value=data["t0"],
            description="Start time (ms):",
            min=0,
            max=1e6,
            **simple_widget_kwargs,
        )
        _make_opt_observers(t0_widget, "t0", drive_widgets, drive_idx)
        tstop_w = BoundedFloatText(
            value=data["tstop"],
            description="Stop time (ms):",
            max=1e6,
            **simple_widget_kwargs,
        )
        _make_opt_observers(tstop_w, "tstop", drive_widgets, drive_idx)
    elif choose_tab_drive_or_opt == "drive":
        amplitudes = dict()
        for cell_type in cell_types:
            amplitude = data[cell_type]["amplitude"]
            amplitudes[cell_type] = BoundedFloatText(
                value=amplitude,
                description=cell_type,
                min=0,
                max=1e6,
                step=0.01,
                **simple_widget_kwargs,
            )
            # Reset the global t0 and stop with values from the 'data' keyword.
            # It should be same across all the cell-types.
            if amplitude > 0:
                t0 = data[cell_type]["t0"]
                tstop = data[cell_type]["tstop"]

        t0_widget = BoundedFloatText(
            value=t0,
            description="Start time",
            min=0,
            max=1e6,
            step=1.0,
            **simple_widget_kwargs,
        )
        tstop_w = BoundedFloatText(
            value=tstop,
            description="Stop time",
            min=-1,
            max=1e6,
            step=1.0,
            **simple_widget_kwargs,
        )

    # Initialize the drive widget dict
    new_drive_widgets = dict(
        type="Tonic",
        name=name,
        t0=t0_widget,
        tstop=tstop_w,
    )

    # Amplitude widgets
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        syn_widgets_dict = {
            "amplitude": {},
        }
        amplitudes_list = []
        for cell_type in cell_types:
            syn_widgets_dict["amplitude"].update(
                _create_opt_widgets_for_drive_var(
                    cell_type,
                    data["amplitude"][cell_type],
                    f"{cell_type}:",
                    syn_type="amplitude",
                    **complex_opt_widget_kwargs,
                )
            )
            amplitudes_list.append(
                _create_hbox_for_opt_var(
                    cell_type,
                    syn_widgets_dict["amplitude"],
                    opt_tab_quad_hbox_layout,
                )
            )
        new_drive_widgets.update(syn_widgets_dict)
        syn_widgets_list = [HTML(value="<b>Amplitude (nA)</b>")] + amplitudes_list
    elif choose_tab_drive_or_opt == "drive":
        new_drive_widgets["amplitude"] = amplitudes
        widgets_dict = {
            "amplitude": amplitudes,
            "t0": t0_widget,
            "tstop": tstop_w,
        }
        new_drive_widgets.update(widgets_dict)
        syn_widgets_list = (
            [HTML(value="<b>Times (ms):</b>")]
            + [t0_widget, tstop_w]
            + [HTML(value="<b>Amplitude (nA):</b>")]
            + list(amplitudes.values())
        )

    # Finally, decide the "VBox" positioning of all of the above widgets
    # --------------------------------------------------------------------------
    if choose_tab_drive_or_opt == "opt":
        new_drive_box = VBox(
            [
                opt_tab_column_titles,
                HTML(value="<b>Times (ms):</b>"),
                t0_widget,
                tstop_w,
            ]
            + syn_widgets_list
        )
    elif choose_tab_drive_or_opt == "drive":
        new_drive_box = VBox(syn_widgets_list)

    return new_drive_widgets, new_drive_box


def _create_widgets_for_drive(
    drive_type,
    name,
    tstop_widget,
    location,
    choose_tab_drive_or_opt,
    drive_data,
    weights_ampa,
    weights_nmda,
    delays,
    n_drive_cells,
    cell_specific,
    # Drive-tab-specific kwargs
    drive_tab_var_layout=None,
    drive_tab_var_style=None,
    # Optimization-tab-specific kwargs
    opt_tab_quad_hbox_layout=None,
    opt_tab_column_titles=None,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """Build & arrange all widgets for a single Drive.

    When ``choose_tab_drive_or_opt=="drive"``, this creates the widgets for the Drives
    tab. When ``choose_tab_drive_or_opt=="opt"``, this creates the Optimization widgets
    with constraint controls and observers that mirror the Drives-tab widgets.

    Returns ``(new_drive_widgets, new_drive_box)`` — the widget dict and VBox layout.
    """
    if choose_tab_drive_or_opt == "opt":
        kwargs = dict(
            opt_tab_quad_hbox_layout=opt_tab_quad_hbox_layout,
            opt_tab_column_titles=opt_tab_column_titles,
            initial_constraint_range_percentage=initial_constraint_range_percentage,
            opt_tab_var_layout=opt_tab_var_layout,
            opt_tab_var_style=opt_tab_var_style,
            opt_tab_checkbox_layout=opt_tab_checkbox_layout,
            opt_tab_checkbox_style=opt_tab_checkbox_style,
            opt_tab_minmax_layout=opt_tab_minmax_layout,
            opt_tab_minmax_style=opt_tab_minmax_style,
            drive_idx=drive_idx,
            drive_widgets=drive_widgets,
            prior_opt_widget_values=prior_opt_widget_values,
        )
    elif choose_tab_drive_or_opt == "drive":
        kwargs = dict(
            drive_tab_var_layout=drive_tab_var_layout,
            drive_tab_var_style=drive_tab_var_style,
        )
    else:
        raise ValueError(
            f"Invalid value for choose_tab_drive_or_opt: {choose_tab_drive_or_opt}"
        )

    if drive_type in ("Rhythmic", "Bursty"):
        new_drive_widgets, new_drive_box = _create_widgets_for_rhythmic(
            name,
            tstop_widget,
            location,
            choose_tab_drive_or_opt,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
            **kwargs,
        )
    elif drive_type == "Poisson":
        new_drive_widgets, new_drive_box = _create_widgets_for_poisson(
            name,
            tstop_widget,
            location,
            choose_tab_drive_or_opt,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
            **kwargs,
        )
    elif drive_type in ("Evoked", "Gaussian"):
        new_drive_widgets, new_drive_box = _create_widgets_for_evoked(
            name,
            location,
            choose_tab_drive_or_opt,
            data=drive_data,
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            delays=delays,
            n_drive_cells=n_drive_cells,
            cell_specific=cell_specific,
            **kwargs,
        )
    elif drive_type == "Tonic":
        new_drive_widgets, new_drive_box = _create_widgets_for_tonic(
            name,
            tstop_widget,
            choose_tab_drive_or_opt,
            data=drive_data,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown drive type {drive_type}")

    return new_drive_widgets, new_drive_box


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


def _check_if_tonic_bias_exists(drive_widgets):
    for drive in drive_widgets:
        if drive["type"] == "Tonic":
            return False
    return True


def _make_opt_observers(var_widget, var_key, drive_widgets, drive_idx, syn_type=None):
    """Create & set an 'observe' handler for an Optimization widget to a Drive widget.

    Cyclomatic complexity = To infinity, and beyond!
    """

    # Once again, let's use closures to make new observer handlers for every variable,
    # so that the Optimization tab copy of each variable will be auto-updated when the
    # variable is changed in the Drives tab.
    def _make_update_var_func(var_widget, source_widget):
        def _update_var_func(change):
            var_widget.value = source_widget.value

        return _update_var_func

    if syn_type:
        _fn1 = _make_update_var_func(
            var_widget,
            drive_widgets[drive_idx][syn_type][var_key],
        )
        _fn2 = _make_update_var_func(
            drive_widgets[drive_idx][syn_type][var_key],
            var_widget,
        )
        drive_widgets[drive_idx][syn_type][var_key].observe(_fn1, names="value")
        var_widget.observe(_fn2, names="value")
    else:
        _fn1 = _make_update_var_func(
            var_widget,
            drive_widgets[drive_idx][var_key],
        )
        _fn2 = _make_update_var_func(
            drive_widgets[drive_idx][var_key],
            var_widget,
        )
        drive_widgets[drive_idx][var_key].observe(_fn1, names="value")
        var_widget.observe(_fn2, names="value")


def _create_opt_widgets_for_drive_var(
    var_name,
    initial_value,
    var_description,
    syn_type=None,
    init_bool=False,
    initial_constraint_range_percentage=None,
    opt_tab_var_layout=None,
    opt_tab_var_style=None,
    opt_tab_checkbox_layout=None,
    opt_tab_checkbox_style=None,
    opt_tab_minmax_layout=None,
    opt_tab_minmax_style=None,
    drive_idx=None,
    drive_widgets=None,
    prior_opt_widget_values=None,
):
    """For a drive variable, create its multiple Optimization widgets and observers.

    For each drive's variable that we want to allow Optimization for, we need to create 4 widgets.
    If the user has previously indicated (by checking a checkbox) that they want to optimize this
    variable, then the widget will be initialized with those prior values. The four widgets are:

        1. The widget showing that variable inside the drive's accordion entry. This
        widget is disabled/ghosted by default, since contains the same information as the variable's
        value in the equivalent widget in the Drives tab. Towards the end of this function, we
        create an observation such that the Optimization version of the widget updates its value
        based on changes in the Drive version of the widget.

        2. A checkbox widget for whether this variable should have its value and
        constraints used during the Optimization. This checkbox is False by default, and is used
        later in `_generate_constraints_and_func` to build the "parameters update function"
        (`set_params`) that is needed by the Optimization process.

        3. A widget for the "minimum" percentage of the current value of the drive
        variable, to be used as the minimum of the constraint range. This will only be actually used
        if the checkbox is checked.

        4. A widget for the "maximum" percentage of the current value of the drive
        variable, to be used as the maximum of the constraint range. This will only be actually used
        if the checkbox is checked.
    """

    # These variables will hold prior Optimization widget values, if they exist. This way, if the
    # user has previously executed GUI Optimization runs and set certain variables to be optimized
    # with specific constraint ranges, those values will be retained and used to initialize the
    # widgets below.
    prior_checkbox, prior_min_pct, prior_max_pct = None, None, None
    if prior_opt_widget_values:
        # Create the unique parameter name for this variable in this drive, IF it exists in the keys
        # of `prior_opt_widget_values`. If it does not exist, then this var is None.
        unique_param_name = _name_check(
            var_name,
            drive_widgets[drive_idx],
            prior_opt_widget_values,
            syn_type,
        )
        if unique_param_name:
            prior_checkbox = True
            prior_min_pct = prior_opt_widget_values[unique_param_name][0]
            prior_max_pct = prior_opt_widget_values[unique_param_name][1]

    var_widget = BoundedFloatText(
        value=initial_value,
        description=var_description,
        min=0,
        max=1e6,
        step=0.01,
        layout=opt_tab_var_layout,
        style=opt_tab_var_style,
    )
    opt_checkbox_widget = Checkbox(
        value=(prior_checkbox if prior_checkbox else init_bool),
        layout=opt_tab_checkbox_layout,
        style=opt_tab_checkbox_style,
    )
    opt_min_widget = BoundedFloatText(
        value=(
            prior_min_pct
            if prior_min_pct is not None
            else (100 - initial_constraint_range_percentage)
        ),
        description="Min:",
        min=0,
        max=100,
        step=1,
        layout=opt_tab_minmax_layout,
        style=opt_tab_minmax_style,
    )
    opt_max_widget = BoundedFloatText(
        value=(
            prior_max_pct
            if prior_max_pct is not None
            else (100 + initial_constraint_range_percentage)
        ),
        description="Max:",
        min=100,
        max=1000,
        step=1,
        layout=opt_tab_minmax_layout,
        style=opt_tab_minmax_style,
    )

    # Connect the main var_widget to its observed Drives tab equivalent, and vice-versa
    _make_opt_observers(var_widget, var_name, drive_widgets, drive_idx, syn_type)

    return {
        f"{var_name}": var_widget,
        f"{var_name}_opt_checkbox": opt_checkbox_widget,
        f"{var_name}_opt_min": opt_min_widget,
        f"{var_name}_opt_max": opt_max_widget,
    }


def _create_hbox_for_opt_var(var_name, widget_dict, layout):
    """Helper function for placement & layout of a single drive variable's Opt widgets."""
    return HBox(
        [
            widget_dict[f"{var_name}"],
            widget_dict[f"{var_name}_opt_checkbox"],
            widget_dict[f"{var_name}_opt_min"],
            widget_dict[f"{var_name}_opt_max"],
        ],
        layout=layout,
    )


def _build_constraints(drive, syn_type=None, apply_percentages=False):
    """Build a dictionary containing parameter constraint values or percentages.

    Parameters
    ----------
    drive : dict
        Dictionary containing drive parameter widgets and their values.
    syn_type : str, optional
        The synapse type to build constraints for (e.g., 'ampa', 'nmda').
    apply_percentages : bool, default=False
        If True, converts percentage values to actual constraint values based on
        current parameter values. If False, returns raw percentage values.

    Returns
    -------
    output_constraints : dict
        Dictionary mapping unique parameter names to constraint tuples. Keys are
        formatted as '{drive_type}_{drive_name}_{syn_type}_{var_name}' (or without
        syn_type if not provided). Values are tuples of (min_value, max_value),
        either as actual values or percentages depending on `apply_percentages`.
    """
    output_constraints = {}
    if syn_type:
        input_dict = drive[syn_type]
    else:
        input_dict = drive
    for key in input_dict.keys():
        # For every variable with a checkbox, but only if the checkbox
        # is true/checked
        if ("_opt_checkbox" in key) and (input_dict[key].value):
            # Extract the var name
            var_name = key.split("_opt_checkbox")[0]
            # Create a new, unique var name for this drive's instance of
            # that variable, which will become our key in our
            # `input_constraints` dict. This name will be used by `name_check`.
            unique_param_name = str(
                drive["type"]
                + "_"
                + drive["name"]
                + "_"
                + (syn_type + "_" if syn_type else "")
                + var_name
            )
            # Get the percentage values from the min/max widgets (raw percentages)
            min_pct = input_dict[var_name + "_opt_min"].value
            max_pct = input_dict[var_name + "_opt_max"].value
            if apply_percentages:
                # Get the current value of the parameter
                current_value = input_dict[var_name].value
                # Convert percentages to actual values
                # e.g., if current_value=10 and min_pct=50 (%),
                #       then min_value = 10*(50/100) = 5
                min_value = current_value * (min_pct / 100)
                max_value = current_value * (max_pct / 100)
                # Use the unique name as the key, and add the bounds as actual values
                output_constraints.update(
                    {unique_param_name: tuple([min_value, max_value])}
                )
            else:
                output_constraints.update(
                    {unique_param_name: tuple([min_pct, max_pct])}
                )
    return output_constraints


def _name_check(var, drive, params, syn_type=None):
    """Check if a unique parameter name exists in the input `params` dictionary.

    This function constructs a unique parameter name from `drive` metadata and
    checks if it exists in the provided `params` dictionary.

    Parameters
    ----------
    var : str
        The variable name to check.
    drive : dict
        Dictionary containing metadata and widgets for a specific drive.
    params : dict
        Dictionary of parameters to search in (usually a dictionary of prior widget values).
    syn_type : str, optional
        Synapse type to include in the parameter name, if any.

    Returns
    -------
    str or None
        The unique parameter name if it exists in `params`, None otherwise.
    """
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


def _use_nonsyn_params_if_exists(var_name, drive, params):
    """Retrieve a non-synaptice parameter value from `params` (if it exists), else from `drive`.

    Parameters
    ----------
    var_name : str
        The full, unique variable name to look for.
    drive : dict
        Dictionary containing metadata and widgets for a specific drive.
    params : dict
        Dictionary of "prior" parameters to check first for the variable.

    Returns
    -------
    value : various
        The parameter value from params dict if the name-checked key exists, otherwise the value
        from the drive dict's `value` of the widget for `var_name`.
    """
    return (
        params[_name_check(var_name, drive, params)]
        if _name_check(var_name, drive, params)
        else drive[var_name].value
    )


def _create_parametrized_syn_dicts_if_exist(syn_type, drive, params):
    """Create dict of synaptic parameters if they exist in `params`, else use values from `drive`.

    This function creates a dictionary of synaptic parameters for different cell types, checking if
    parametrized versions exist in the `params` dict. If a parametrized version exists, it uses that
    value; otherwise, it falls back to the drive's default value. In the special case of
    'amplitude', it also excludes the `L5_basket` cell type if the drive location is 'distal'.

    Parameters
    ----------
    syn_type : {"weights_ampa", "weights_nmda", "delays", "rate_constant", "amplitude"}
        The type of synaptic parameter to create the dictionary for.
    drive : dict
        Dictionary containing metadata and widgets for a specific drive.
    params : dict
        Dictionary of "prior" parameters that may contain parametrized versions of synaptic
        parameters.

    Returns
    -------
    output_dict : dict
        A nested dictionary with `syn_type` as the outer key and cell types as inner keys, each
        mapping to their respective values for the parameter inherent to `syn_type`.
    """
    cell_types = [
        "L5_pyramidal",
        "L2_pyramidal",
        "L5_basket",
        "L2_basket",
    ]
    if syn_type != "amplitude":
        if drive["location"] == "distal":
            cell_types.remove("L5_basket")
    output_dict = {syn_type: {}}
    for ct in cell_types:
        output_dict[syn_type].update(
            {
                ct: (
                    params[_name_check(ct, drive, params, syn_type)]
                    if _name_check(ct, drive, params, syn_type)
                    else drive[syn_type][ct].value
                )
            }
        )
    return output_dict


def _generate_constraints_and_func(net, opt_drive_widgets):
    """Dynamically create a constraints dict and usage/update function from Opt widgets.

    Using the widgets of both the Drive and Optimization tabs, this does two things:

        1. Creates a "constraints" dictionary where, for any drive's variable in the
        Optimization tab, if that variable's "opt_checkbox" widget is checked, a
        key-value pair is created. The key is a unique name for that variable that
        includes the drive type, drive name, synaptic variable type if present, and
        variable name. The value is a tuple of the minimum and maximum constraint values
        from the appropriate Optimization widgets for that variable.

        2. Creates a "set_params" function consumes the "constraints" dictionary. This
        function creates ALL drives for the assumed-drive-less Network object. For the
        variables of each drive, if there is a name-matching entry in the "constraints"
        dictionary, then that value is used (enabling the Optimizer to control and
        vary/"optimize" that variable). If there is no name-matching entry in the
        constraints dictionary, then the value of the Drive widget for that variable is
        used.

    Note that THIS is where the Drives are actually added to the Network object during
    Optimization.

    This was originally created from the code in `_init_network_from_widgets`, but it
    has no proper equivalent for the Drives. This one took some brainpower to make.
    """
    # First, iterate through set of variable-specific widgets for each drive, assemble
    # param var names, and grab constraint values for those whose checkbox is true. This
    # builds a `constraints` dictionary that is FLAT, where the keys are long variable
    # names (with their context) for which the user has checked the checkbox, and their
    # values are a tuple with their min and max constraints.
    # ------------------------------------------------------------------------------
    constraints = {}
    for drive in opt_drive_widgets:
        if drive["type"] in ("Tonic"):
            constraints.update(_build_constraints(drive, apply_percentages=True))
            constraints.update(
                _build_constraints(drive, syn_type="amplitude", apply_percentages=True)
            )
        else:
            # Synaptic variables are a special case, since they are dicts instead of
            # single values
            for syn_type in ("weights_ampa", "weights_nmda", "delays"):
                constraints.update(_build_constraints(drive, syn_type=syn_type))
            if drive["type"] == "Poisson":
                constraints.update(_build_constraints(drive, apply_percentages=True))
                constraints.update(
                    _build_constraints(
                        drive,
                        syn_type="rate_constant",
                        apply_percentages=True,
                    )
                )
            elif drive["type"] in ("Evoked", "Gaussian"):
                constraints.update(_build_constraints(drive, apply_percentages=True))
            elif drive["type"] in ("Rhythmic", "Bursty"):
                constraints.update(_build_constraints(drive, apply_percentages=True))

    # Second, create a new `set_params` function that iterates through the drive
    # widgets AGAIN, but which deploys our newly-created `constraints` dict:
    # ------------------------------------------------------------------------------
    def set_params(net, params):
        for drive in opt_drive_widgets:
            if drive["type"] in ("Tonic"):
                deployed_syn_dicts = {}
                deployed_syn_dicts.update(
                    _create_parametrized_syn_dicts_if_exist("amplitude", drive, params)
                )
                net.add_tonic_bias(
                    amplitude=deployed_syn_dicts["amplitude"],
                    t0=_use_nonsyn_params_if_exists("t0", drive, params),
                    tstop=_use_nonsyn_params_if_exists("tstop", drive, params),
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

                deployed_syn_dicts = {}
                for syn_type in ("weights_ampa", "weights_nmda", "delays"):
                    deployed_syn_dicts.update(
                        _create_parametrized_syn_dicts_if_exist(syn_type, drive, params)
                    )

                print(f"drive type is {drive['type']}, location={drive['location']}")
                if drive["type"] == "Poisson":
                    deployed_syn_dicts.update(
                        _create_parametrized_syn_dicts_if_exist(
                            "rate_constant", drive, params
                        )
                    )

                    net.add_poisson_drive(
                        name=drive["name"],
                        tstart=_use_nonsyn_params_if_exists("tstart", drive, params),
                        tstop=_use_nonsyn_params_if_exists("tstop", drive, params),
                        rate_constant=deployed_syn_dicts["rate_constant"],
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
                        mu=_use_nonsyn_params_if_exists("mu", drive, params),
                        sigma=_use_nonsyn_params_if_exists("sigma", drive, params),
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
                        tstart=_use_nonsyn_params_if_exists("tstart", drive, params),
                        tstart_std=_use_nonsyn_params_if_exists(
                            "tstart_std", drive, params
                        ),
                        tstop=_use_nonsyn_params_if_exists("tstop", drive, params),
                        location=drive["location"],
                        burst_rate=_use_nonsyn_params_if_exists(
                            "burst_rate", drive, params
                        ),
                        burst_std=_use_nonsyn_params_if_exists(
                            "burst_std", drive, params
                        ),
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
    opt_smoothing,
    opt_scaling,
    opt_target_widgets,
):
    """Run an Optimization, then re-run its final simulation and plot its outputs.

    This does the following steps:

    1. Some setup based on the existing simulation data and names.
    2. Perform some input validation based on the type of optimization selected
        (i.e. checks for valid data if the objective function is `dipole_rmse`).
    3. Instantiate the Network object, but NOT including any drives. This creates the
        Network object at `simulation_data[_sim_name]["net"]` specifically.
    4. Dynamically build the `constraints` dict and the function (`set_params_func`)
        used to deploy user-provided constraints, update parameters during each
        optimization iteration, and create the drives anew every time (since we are
        optimizing for drive parameters).
    5. Check that at least some constraints have been indicated by the user.
    6. Create the Optimizer object using our Network, `constraints`, `set_params_func`,
        and other top-level Optimization widget values.
    7. Select and setup our simulation backend.
    8. The big one: execute the actual Optimizer fitting, including checking and
        applying "target" widgets options.
    9. Afterwards, create a new `simulation_data` name (depending on existing names) and
        re-execute the final version of optimized Network. (We have to do this since the
        Optimizer object does not (I think?) give us the final simulation output
        data). The optimized simulation data gets saved to `<current GUI Simulation Name
        widget value>_optimized` or something similar.
    10. Save the re-executed final optimized Network and simulation data into
        `simulation_data`, so that it can be used and explored in the GUI.
    11. Plot the resulting optimized dipole.
    12. Return the "JSON config" string of our optimized Network, which the GUI will use
        (in `HNNGUI._run_opt_button_clicked`) to RE-load all parameters of all drives.
        This way, after a successful optimization, the GUI's Drives' parameters will
        reflect their newly optimized values.

    This was built based off of `run_button_clicked`.
    """
    with log_out:
        # Sim data setup (and related input validation)
        # ------------------------------------------------------------------------------
        simulation_data = all_data["simulation_data"]

        # clear empty trash simulations
        #
        # AES: a "trash" simulation appears to be created (named "default") even if all
        # a user does is load an external dipole data file. However, I do not fully
        # understand how VizManager et al. manages the simulation data (I find it very
        # confusing) so I am NOT touching it.
        for _name in tuple(simulation_data.keys()):
            if len(simulation_data[_name]["dpls"]) == 0:
                del simulation_data[_name]

        _sim_name = widget_simulation_name.value

        # RMSE Target data extraction (and related input validation)
        # ------------------------------------------------------------------------------
        if opt_obj_fun == "dipole_rmse":
            opt_rmse_target_data_name = opt_target_widgets["rmse_target_data"].value
            if not opt_rmse_target_data_name:
                # In this case, they probably have not run any simulations or loaded any
                # data.
                logger.error(
                    textwrap.dedent("""
                    You have not selected a dataset to use as the target of
                    optimization. Please load and select a dataset of dipole data to
                    optimize towards.
                    """).replace("\n", " ")
                )
                simulation_status_bar.value = simulation_status_contents["failed"]
                return None
            elif (opt_rmse_target_data_name == "default") and (
                not simulation_data["default"]["dpls"]
            ):
                # In this case, they have selected "default", which is the default name
                # of the first simulation, BUT they have not actually run any
                # simulations yet. They likely either want to compare against a
                # simulation result, or (more likely) forgot to load their experimental
                # target data first.
                #
                # ATTN: How we want to handle this, and what we want to communicate,
                # needs some discussion and thinking.
                logger.error(
                    textwrap.dedent("""
                    You have selected the 'default' dataset to use as the target of
                    optimization, but there is no dipole data associated with that
                    dataset.  Please either load and select a dataset of dipole data to
                    optimize towards, or run a simulation first if you want to optimize
                    against that simulation.
                    """).replace("\n", " ")
                )
                simulation_status_bar.value = simulation_status_contents["failed"]
                return None
            else:
                # Extract the actual target data
                # Like everywhere else in the GUI, we only support usage of single-trial
                # dipole data.
                target_dipole = simulation_data[opt_rmse_target_data_name]["dpls"][0]

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
        set_params_func, constraints = _generate_constraints_and_func(
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
            return None

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
            # Store PSD parameters for history tracking
            psd_f_bands = None
            psd_relative_bandpower = None

            try:
                if opt_obj_fun == "dipole_rmse":
                    optim.fit(
                        target=target_dipole,
                        n_trials=opt_target_widgets["n_trials"].value,
                        smooth_window_len=opt_smoothing,
                        scale_factor=opt_scaling,
                    )
                elif opt_obj_fun == "maximize_psd":
                    if opt_target_widgets["psd_target_band2_checkbox"].value:
                        f_bands = [
                            (
                                opt_target_widgets["psd_target_band1_min"].value,
                                opt_target_widgets["psd_target_band1_max"].value,
                            ),
                            (
                                opt_target_widgets["psd_target_band2_min"].value,
                                opt_target_widgets["psd_target_band2_max"].value,
                            ),
                        ]
                        relative_bandpower = [
                            opt_target_widgets["psd_target_band1_proportion"].value,
                            opt_target_widgets["psd_target_band2_proportion"].value,
                        ]
                    else:
                        f_bands = [
                            (
                                opt_target_widgets["psd_target_band1_min"].value,
                                opt_target_widgets["psd_target_band1_max"].value,
                            )
                        ]
                        relative_bandpower = [
                            opt_target_widgets["psd_target_band1_proportion"].value,
                        ]

                    # Store for history tracking
                    psd_f_bands = f_bands
                    psd_relative_bandpower = relative_bandpower

                    optim.fit(
                        f_bands=f_bands,
                        relative_bandpower=relative_bandpower,
                        smooth_window_len=opt_smoothing,
                        scale_factor=opt_scaling,
                    )

            except Exception as e:
                logger.error(
                    f"Optimization fitting failed due to exception: '{e}'",
                    exc_info=True,
                )
                simulation_status_bar.value = simulation_status_contents["failed"]
                raise

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
                n_trials=opt_target_widgets["n_trials"].value,
            )
            # Finally, update the list of simulations to include our new one:
            sim_names = [
                sim_name
                for sim_name in simulation_data
                if simulation_data[sim_name]["net"] is not None
            ]
            simulations_list_widget.options = sim_names
            simulations_list_widget.value = sim_names[-1]

            # Report back to the user, now that all simulations/output are completed:
            logger.info(
                textwrap.dedent(f"""
                Optimization finished!
                Don't forget to "Save Network"!
                First objective function result: {optim.obj_[0]}
                Last objective function result: {optim.obj_[-1]}
                Diff: {abs(optim.obj_[-1] - optim.obj_[0])}
                """)
            )
            # Check if optimization showed ANY difference in the objective function. If
            # it did not, then we made no progress.
            #
            # ATTN: If we want to include something like the following in our automated
            # testing, then we need to be sure (or at least very confident) that the
            # same inputs will produce the same outputs (i.e. that our optimization is
            # deterministic). Is it?
            if np.all(optim.obj_ == optim.obj_[0]):
                logger.warning(
                    textwrap.dedent("""
                    The objective function did not change over the course of the
                    optimization. You probably need to increase the number of max
                    iterations in order to start converging.
                    """).replace("\n", " ")
                )
                simulation_status_bar.value = simulation_status_contents["failed"]
            else:
                simulation_status_bar.value = simulation_status_contents["finished"]

        # ------------------------------------------------------------------------------
        # The remainder of this function is just repeating some post-run visualization
        # steps, which are identical to those in `run_button_clicked`
        viz_manager.reset_fig_config_tabs()

        # update default visualization params in gui based on widget
        fig_default_params["default_smoothing"] = opt_smoothing
        fig_default_params["default_scaling"] = opt_scaling
        fig_default_params["default_min_frequency"] = widget_min_frequency.value
        fig_default_params["default_max_frequency"] = widget_max_frequency.value

        # change default visualization params in viz_manager to mirror gui
        for widget, value in fig_default_params.items():
            viz_manager.fig_default_params[widget] = value

        viz_manager.add_figure()
        fig_name = _idx2figname(viz_manager.data["fig_idx"]["idx"] - 1)
        viz_manager._simulate_edit_figure(
            fig_name,
            "ax0",
            new_name,
            "input histogram",
            {},
            "plot",
        )
        viz_manager._simulate_edit_figure(
            fig_name,
            "ax1",
            new_name,
            "current dipole",
            {
                "data_to_compare": opt_target_widgets["rmse_target_data"].value
                if opt_obj_fun == "dipole_rmse"
                else None
            },
            "plot",
        )

        # ATTN: Maybe force a file-save of the final network params automatically at the
        # end? Since the `HNNGUI.save_config_button` is just a huge HTML element itself,
        # I can't get it to artificially ".click()" to actually initiate a download. Is
        # there even a way to do this?

        # Return both the optimized config and the optimizer results
        optimized_config = serialize_config(all_data, new_name)
        opt_result = {
            "initial_params": optim.initial_params,
            "opt_params": optim.opt_params_,
            "obj_values": optim.obj_,
            "obj_fun": opt_obj_fun,
            "solver": opt_solver,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "max_iter": opt_max_iter,
        }

        # Add target dataset name if using dipole_rmse
        if opt_obj_fun == "dipole_rmse":
            opt_result["target_data"] = opt_target_widgets["rmse_target_data"].value
            opt_result["n_trials"] = opt_target_widgets["n_trials"].value
        # Add frequency band parameters if using maximize_psd
        elif opt_obj_fun == "maximize_psd":
            opt_result["psd_f_bands"] = psd_f_bands
            opt_result["psd_relative_bandpower"] = psd_relative_bandpower
        return optimized_config, opt_result


def launch():
    """Launch voila with hnn_widget.ipynb.

    You can pass voila commandline parameters as usual.
    """
    from voila.app import main

    notebook_path = Path(__file__).parent / "hnn_widget.ipynb"
    main([str(notebook_path.resolve()), *sys.argv[1:]])
