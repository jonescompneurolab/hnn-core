"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import json
import codecs
import os.path as op
from functools import partial, update_wrapper

import numpy as np

import hnn_core
from hnn_core import simulate_dipole, read_params, Network
from hnn_core.params import _read_legacy_params, _read_json

from IPython.display import display

from ipywidgets import (FloatLogSlider, Dropdown, Button, RadioButtons,
                        fixed, interactive_output, interactive, interact,
                        FloatText, BoundedFloatText, IntText, FileUpload,
                        HTML, Output, HBox, VBox, Tab, Accordion,
                        Layout, AppLayout)


def create_expanded_button(description, button_style, height):
    style = {'button_color': '#8A2BE2'}
    return Button(description=description, button_style=button_style,
                  layout=Layout(height=height, width='auto'),
                  style=style)


def _get_sliders(params, param_keys):
    """Get sliders"""
    style = {'description_width': '150px'}
    sliders = list()
    for d in param_keys:
        slider = FloatLogSlider(
            value=params[d], min=-5, max=1, step=0.2,
            description=d.split('gbar_')[1],
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.2e', style=style)
        sliders.append(slider)

    def _update_params(variables, **updates):
        params.update(dict(**updates))

    interactive_output(_update_params, {s.description: s for s in sliders})
    return sliders


def _get_cell_specific_widgets(layout, style):
    kwargs = dict(layout=layout, style=style)
    cell_types = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket',
                  'L2_basket']
    weights_ampa, weights_nmda, delays = dict(), dict(), dict()
    for cell_type in cell_types:
        weights_ampa[f'{cell_type}'] = FloatText(
            value=0., description=f'{cell_type}:', **kwargs)
        weights_nmda[f'{cell_type}'] = FloatText(
            value=0., description=f'{cell_type}:', **kwargs)
        delays[f'{cell_type}'] = FloatText(
            value=0.1, description=f'{cell_type}:', **kwargs)

    widgets_dict = {'weights_ampa': weights_ampa,
                    'weights_nmda': weights_nmda, 'delays': delays}
    widgets_list = ([HTML(value="<b>AMPA weights</b>")] +
                    list(weights_ampa.values()) +
                    [HTML(value="<b>NMDA weights</b>")] +
                    list(weights_nmda.values()) +
                    [HTML(value="<b>Synaptic delays</b>")] +
                    list(delays.values()))
    return widgets_list, widgets_dict


def _get_rhythmic_widget(name, tstop_widget, layout, style):

    kwargs = dict(layout=layout, style=style)
    tstart = FloatText(value=0., description='Start time (s)', **kwargs)
    tstart_std = FloatText(value=0, description='Start time dev (s)',
                           **kwargs)
    tstop = BoundedFloatText(value=tstop_widget.value,
                             description='Stop time (s)',
                             max=tstop_widget.value, **kwargs)
    burst_rate = FloatText(value=7.5, description='Burst rate (Hz)', **kwargs)
    burst_std = FloatText(value=0, description='Burst std dev (Hz)', **kwargs)
    repeats = FloatText(value=1, description='Repeats', **kwargs)
    seedcore = IntText(value=14, description='Seed', **kwargs)
    location = RadioButtons(options=['proximal', 'distal'])

    widgets_list, widgets_dict = _get_cell_specific_widgets(layout, style)
    drive_box = VBox([tstart, tstart_std, tstop, burst_rate, burst_std,
                      repeats, location, seedcore] + widgets_list)
    drive = dict(type='Rhythmic', name=name,
                 tstart=tstart, tstart_std=tstart_std,
                 burst_rate=burst_rate, burst_std=burst_std,
                 repeats=repeats, seedcore=seedcore,
                 location=location, tstop=tstop)
    drive.update(widgets_dict)
    return drive, drive_box


def _get_poisson_widget(name, tstop_widget, layout, style):
    tstart = FloatText(value=0.0, description='Start time (s)',
                       layout=layout, style=style)
    tstop = BoundedFloatText(value=tstop_widget.value,
                             max=tstop_widget.value,
                             description='Stop time (s)',
                             layout=layout, style=style)
    seedcore = IntText(value=14, description='Seed',
                       layout=layout, style=style)
    location = RadioButtons(options=['proximal', 'distal'])

    cell_types = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket',
                  'L2_basket']
    rate_constant = dict()
    for cell_type in cell_types:
        rate_constant[f'{cell_type}'] = FloatText(
            value=8.5, description=f'{cell_type}:',
            layout=layout, style=style)

    widgets_list, widgets_dict = _get_cell_specific_widgets(layout, style)
    widgets_dict.update({'rate_constant': rate_constant})
    widgets_list.extend([HTML(value="<b>Rate constants</b>")] +
                        list(widgets_dict['rate_constant'].values()))

    drive_box = VBox([tstart, tstop, seedcore, location] + widgets_list)
    drive = dict(type='Poisson', name=name, tstart=tstart,
                 tstop=tstop, rate_constant=rate_constant,
                 seedcore=seedcore, location=location)
    drive.update(widgets_dict)
    return drive, drive_box


def _get_evoked_widget(name, layout, style):
    kwargs = dict(layout=layout, style=style)
    mu = FloatText(value=0, description='Mean time:', **kwargs)
    sigma = FloatText(value=1, description='Std dev time:',
                      **kwargs)
    numspikes = IntText(value=1, description='No. Spikes:',
                        **kwargs)
    seedcore = IntText(value=14, description='Seed: ', **kwargs)
    location = RadioButtons(options=['proximal', 'distal'])

    widgets_list, widgets_dict = _get_cell_specific_widgets(layout, style)
    drive_box = VBox([mu, sigma, numspikes, seedcore, location] +
                     widgets_list)
    drive = dict(type='Evoked', name=name,
                 mu=mu, sigma=sigma, numspikes=numspikes,
                 seedcore=seedcore, location=location,
                 sync_within_trial=False)
    drive.update(widgets_dict)
    return drive, drive_box


def add_drive_widget(drive_type, drive_boxes, drive_widgets,
                     drives_out, tstop_widget):
    """Add a widget for a new drive."""
    layout = Layout(width='270px', height='auto')
    style = {'description_width': '150px'}
    drives_out.clear_output()
    with drives_out:
        name = drive_type['new'] + str(len(drive_boxes))

        if drive_type['new'] == 'Rhythmic':
            drive, drive_box = _get_rhythmic_widget(name, tstop_widget,
                                                    layout, style)
        elif drive_type['new'] == 'Poisson':
            drive, drive_box = _get_poisson_widget(name, tstop_widget,
                                                   layout, style)
        elif drive_type['new'] == 'Evoked':
            drive, drive_box = _get_evoked_widget(name, layout, style)

        if drive_type['new'] in ['Evoked', 'Poisson', 'Rhythmic']:
            drive_boxes.append(drive_box)
            drive_widgets.append(drive)

        accordion = Accordion(children=drive_boxes,
                              selected_index=len(drive_boxes) - 1)
        for idx, drive in enumerate(drive_widgets):
            accordion.set_title(idx, drive['name'])
        display(accordion)


def update_plot_window(variables, plot_out, plot_type):
    plot_out.clear_output()

    if not (plot_type['type'] == 'change' and plot_type['name'] == 'value'):
        return

    with plot_out:
        if plot_type['new'] == 'spikes':
            variables['net'].cell_response.plot_spikes_raster()
        elif plot_type['new'] == 'current dipole':
            variables['dpls'][0].plot()
        elif plot_type['new'] == 'input histogram':
            variables['net'].cell_response.plot_spikes_hist()
        elif plot_type['new'] == 'PSD':
            variables['dpls'][0].plot_psd(fmin=0, fmax=50)
        elif plot_type['new'] == 'spectogram':
            freqs = np.arange(10., 100., 1.)
            n_cycles = freqs / 8.
            variables['dpls'][0].plot_tfr_morlet(freqs, n_cycles=n_cycles)
        elif plot_type['new'] == 'network':
            variables['net'].plot_cells()


def on_upload_change(change, sliders, params):
    if len(change['owner'].value) == 0:
        return

    params_fname = change['owner'].metadata[0]['name']
    file_uploaded = change['owner'].value
    param_data = list(file_uploaded.values())[0]['content']
    param_data = codecs.decode(param_data, encoding="utf-8")

    ext = op.splitext(params_fname)[1]
    read_func = {'.json': _read_json, '.param': _read_legacy_params}
    params_network = read_func[ext](param_data)

    for slider in sliders:
        for sl in slider:
            key = 'gbar_' + sl.description
            sl.value = params_network[key]

    params.update(params_network)


def run_button_clicked(log_out, plot_out, drive_widgets, variables, tstep,
                       tstop, params, b):
    """Run the simulation and plot outputs."""
    plot_out.clear_output()
    log_out.clear_output()
    with log_out:
        params['dt'] = tstep.value
        params['tstop'] = tstop.value
        variables['net'] = Network(params, add_drives_from_params=False)
    for drive in drive_widgets:
        weights_ampa = {k: v.value for k, v in drive['weights_ampa'].items()}
        weights_nmda = {k: v.value for k, v in drive['weights_nmda'].items()}
        synaptic_delays = {k: v.value for k, v in drive['delays'].items()}
        if drive['type'] == 'Poisson':
            rate_constant = {k: v.value for k, v in
                             drive['rate_constant'].items() if v.value > 0}
            weights_ampa = {k: v for k, v in weights_ampa.items() if k in
                            rate_constant}
            weights_nmda = {k: v for k, v in weights_nmda.items() if k in
                            rate_constant}
            variables['net'].add_poisson_drive(
                name=drive['name'],
                tstart=drive['tstart'].value,
                tstop=drive['tstop'].value,
                rate_constant=rate_constant,
                location=drive['location'].value,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                synaptic_delays=synaptic_delays,
                space_constant=100.0,
                seedcore=drive['seedcore'].value
            )
        elif drive['type'] == 'Evoked':
            variables['net'].add_evoked_drive(
                name=drive['name'],
                mu=drive['mu'].value,
                sigma=drive['sigma'].value,
                numspikes=drive['numspikes'].value,
                sync_within_trial=False,
                location=drive['location'].value,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                synaptic_delays=synaptic_delays,
                space_constant=3.0,
                seedcore=drive['seedcore'].value
            )
        elif drive['type'] == 'Rhythmic':
            variables['net'].add_bursty_drive(
                name=drive['name'],
                tstart=drive['tstart'].value,
                tstart_std=drive['tstart_std'].value,
                burst_rate=drive['burst_rate'].value,
                burst_std=drive['burst_std'].value,
                repeats=drive['repeats'].value,
                location=drive['location'].value,
                tstop=drive['tstop'].value,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                synaptic_delays=synaptic_delays,
                seedcore=drive['seedcore'].value
            )
    with log_out:
        variables['dpls'] = simulate_dipole(variables['net'], n_trials=1)
    with plot_out:
        variables['dpls'][0].plot()


def run_hnn_gui():
    """Create the HNN GUI."""

    hnn_core_root = op.join(op.dirname(hnn_core.__file__))

    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    drive_widgets = list()
    drive_boxes = list()
    variables = dict(net=None, dpls=None)

    def _add_drive_widget(drive_type):
        return add_drive_widget(drive_type, drive_boxes,
                                drive_widgets, drives_out, tstop)

    def _run_button_clicked(b):
        return run_button_clicked(log_out, plot_out, drive_widgets, variables,
                                  tstep, tstop, params, b)

    def _on_upload_change(change):
        return log_out.capture(on_upload_change(change, sliders, params))

    def _update_plot_window(plot_type):
        return update_plot_window(variables, plot_out, plot_type)

    def _delete_drives_clicked(b):
        drives_out.clear_output()
        # black magic: the following does not work
        # global drive_widgets; drive_widgets = list()
        while len(drive_widgets) > 0:
            drive_widgets.pop()
            drive_boxes.pop()

    # Output windows
    drives_out = Output()  # window to add new drives
    log_out = Output(layout={'border': '1px solid gray', 'height': '150px',
                             'overflow_y': 'auto'})
    plot_out = Output(layout={'border': '1px solid gray', 'height': '350px'})

    # header_button
    header_button = create_expanded_button('HUMAN NEOCORTICAL NEUROSOLVER',
                                           'success', height='40px')

    # Simulation parameters
    tstop = FloatText(value=170, description='tstop (s):', disabled=False)
    tstep = FloatText(value=0.025, description='tstep (s):', disabled=False)
    simulation_box = VBox([tstop, tstep])

    # Sliders to change local-connectivity params
    sliders = [_get_sliders(params,
               ['gbar_L2Pyr_L2Pyr_ampa', 'gbar_L2Pyr_L2Pyr_nmda',
                'gbar_L2Basket_L2Pyr_gabaa', 'gbar_L2Basket_L2Pyr_gabab']),
               _get_sliders(params,
               ['gbar_L2Pyr_L5Pyr', 'gbar_L2Basket_L5Pyr',
                'gbar_L5Pyr_L5Pyr_ampa', 'gbar_L5Pyr_L5Pyr_nmda',
                'gbar_L5Basket_L5Pyr_gabaa', 'gbar_L5Basket_L5Pyr_gabab']),
               _get_sliders(params,
               ['gbar_L2Pyr_L2Basket', 'gbar_L2Basket_L2Basket']),
               _get_sliders(params,
               ['gbar_L2Pyr_L5Pyr', 'gbar_L2Basket_L5Pyr'])]

    # accordians to group local-connectivity by cell type
    boxes = [VBox(slider) for slider in sliders]
    titles = ['Layer 2/3 Pyramidal', 'Layer 5 Pyramidal', 'Layer 2 Basket',
              'Layer 5 Basket']
    accordian = Accordion(children=boxes)
    for idx, title in enumerate(titles):
        accordian.set_title(idx, title)

    # Dropdown for different drives
    layout = Layout(width='200px', height='auto')
    drives_dropdown = Dropdown(
        options=['Evoked', 'Poisson', 'Rhythmic', ''],
        value='', description='Drive:', disabled=False,
        layout=layout)

    # XXX: should be simpler to use Stacked class starting
    # from IPywidgets > 8.0
    interactive(_add_drive_widget, drive_type='Evoked')
    drives_dropdown.observe(_add_drive_widget, 'value')
    drives_options = VBox([drives_dropdown, drives_out])

    # Tabs for left pane
    left_tab = Tab()
    left_tab.children = [simulation_box, accordian, drives_options]
    titles = ['Simulation', 'Cell connectivity', 'Drives']
    for idx, title in enumerate(titles):
        left_tab.set_title(idx, title)

    # Dropdown menu to switch between plots
    plot_dropdown = Dropdown(
        options=['input histogram', 'current dipole',
                 'spikes', 'PSD', 'spectogram', 'network'],
        value='current dipole', description='Plot:',
        disabled=False)
    interactive(_update_plot_window, plot_type='current dipole')
    plot_dropdown.observe(_update_plot_window, 'value')

    # Run, delete drives and load button
    run_button = create_expanded_button('Run', 'success', height='30px')
    style = {'button_color': '#8A2BE2', 'font_color': 'white'}
    load_button = FileUpload(accept='.json,.param', multiple=False,
                             style=style, description='Load network',
                             button_style='success')
    delete_button = create_expanded_button('Delete drives', 'success', height='30px')

    load_button.observe(_on_upload_change)
    run_button.on_click(_run_button_clicked)
    delete_button.on_click(_delete_drives_clicked)
    footer = HBox([run_button, load_button, delete_button, plot_dropdown])

    right_sidebar = VBox([plot_out, log_out])

    # Final layout of the app
    hnn_gui = AppLayout(header=header_button,
                        left_sidebar=left_tab,
                        right_sidebar=right_sidebar,
                        footer=footer,
                        pane_widths=['380px', 1, 1],
                        pane_heights=[1, '500px', 1])
    return hnn_gui
