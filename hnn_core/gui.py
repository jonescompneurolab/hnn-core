"""IPywidgets GUI."""
# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import os.path as op
from functools import partial, update_wrapper

import numpy as np

import hnn_core
from hnn_core import simulate_dipole, read_params, Network

from IPython.display import display

from ipywidgets import (FloatLogSlider, Dropdown, Button, RadioButtons,
                        fixed, interactive_output, interactive, interact,
                        FloatText, FileUpload, HTML, Output,
                        HBox, VBox, Tab, Accordion,
                        Layout, AppLayout)

drives_out = Output()
drive_widgets = list()
drive_titles = list()
drive_boxes = list()


def create_expanded_button(description, button_style, height):
    style = {'button_color': '#8A2BE2'}
    return Button(description=description, button_style=button_style,
                  layout=Layout(height=height, width='auto'),
                  style=style)


def update_params(params, **updates):
    params.update(dict(**updates))
    return params


def _get_sliders(params, param_keys):
    """Get sliders"""
    style = {'description_width': '150px'}
    sliders = list()
    for d in param_keys:
        slider = FloatLogSlider(
            value=params[d], min=-5, max=1, step=0.2,
            description=d.split('gbar_')[1],
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.2e',
            style=style)
        sliders.append(slider)

    _update_params = partial(update_params, params)
    interactive_output(update_params, {s.description: s for s in sliders})
    return sliders


def _get_rhythmic_widget(drive_title, layout, style):
    tstart = FloatText(value=7.5, description='Start time:',
                       layout=layout, style=style)
    tstart_std = FloatText(value=7.5, description='Start time dev:',
                           layout=layout, style=style)
    tstop = FloatText(value=7.5, description='Stop time:',
                      layout=layout, style=style)
    burst_rate = FloatText(value=7.5, description='Burst rate:',
                           layout=layout, style=style)
    burst_std = FloatText(value=7.5, description='Burst std dev:',
                          layout=layout, style=style)
    location = RadioButtons(options=['proximal', 'distal'])

    drive_box = VBox([tstart, tstart_std, tstop, burst_rate, burst_std,
                     location])
    drive = dict(type='Rhythmic', name=drive_title,
                 tstart=tstart, tstart_std=tstart_std,
                 burst_rate=burst_rate, burst_std=burst_std,
                 location=location)
    return drive, drive_box


def _get_poisson_widget(drive_title, layout, style):
    cell_types = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket',
                  'L2_basket']
    tstart = FloatText(value=0.0, description='Start time:',
                       layout=layout, style=style)
    tstop = FloatText(value=8.5, description='Stop time:',
                      layout=layout, style=style)
    location = RadioButtons(options=['proximal', 'distal'])
    labels = {'rate_constant': HTML(value="<b>Rate constants</b>"),
              'ampa': HTML(value="<b>AMPA weights</b>"),
              'nmda': HTML(value="<b>NMDA weights</b>")}
    weights_ampa, weights_nmda = dict(), dict()
    rate_constant = dict()
    for cell_type in cell_types:
        rate_constant[f'{cell_type}'] = FloatText(
            value=8.5, description=f'{cell_type}:',
            layout=layout, style=style)
        weights_ampa[f'{cell_type}'] = FloatText(
            value=0., description=f'{cell_type}:',
            layout=layout, style=style)
        weights_nmda[f'{cell_type}'] = FloatText(
            value=0., description=f'{cell_type}:', layout=layout,
            style=style)

    drive_box = VBox([tstart, tstop, location] + [labels['rate_constant']] +
                     list(rate_constant.values()) + [labels['ampa']] +
                     list(weights_ampa.values()) + [labels['nmda']] +
                     list(weights_nmda.values()))
    drive = dict(type='Poisson', name=drive_title, tstart=tstart,
                 tstop=tstop, rate_constant=rate_constant,
                 location=location, weights_ampa=weights_ampa,
                 weights_nmda=weights_nmda)
    return drive, drive_box


def _get_evoked_widget(drive_title, layout, style):
    mu = FloatText(value=7.5, description='Mean time:',
                   layout=layout)
    sigma = FloatText(value=8.5, description='Std dev time:',
                      layout=layout)
    numspikes = FloatText(value=8.5, description='Number of spikes:',
                          layout=layout)
    location = RadioButtons(options=['proximal', 'distal'])
    weights_ampa_L5Pyr = FloatText(value=8.5,
                                   description='AMPA (L5 Pyr):',
                                   layout=layout)
    weights_nmda_L5Pyr = FloatText(value=8.5,
                                   description='NMDA (L5 Pyr):',
                                   layout=layout)

    drive_box = VBox([mu, sigma, numspikes, location,
                      weights_ampa_L5Pyr, weights_nmda_L5Pyr])
    drive = dict(type='Evoked', name=drive_title,
                 mu=mu, sigma=sigma, numspikes=numspikes,
                 sync_within_trial=False, location=location,
                 weights_ampa_L5Pyr=weights_ampa_L5Pyr,
                 weights_nmda_L5Pyr=weights_nmda_L5Pyr,
                 space_constant=3.0)
    return drive, drive_box


def add_drive_widget(drive_type):
    """Add a widget for a new drive."""
    layout = Layout(width='270px', height='auto')
    style = {'description_width': '150px'}
    drives_out.clear_output()
    with drives_out:
        drive_title = drive_type['new'] + str(len(drive_boxes))

        if drive_type['new'] == 'Rhythmic':
            drive, drive_box = _get_rhythmic_widget(drive_title, layout, style)
        elif drive_type['new'] == 'Poisson':
            drive, drive_box = _get_poisson_widget(drive_title, layout, style)
        elif drive_type['new'] == 'Evoked':
            drive, drive_box = _get_evoked_widget(drive_title, layout, style)

        if drive_type['new'] in ['Evoked', 'Poisson', 'Rhythmic']:
            drive_titles.append(drive_title)
            drive_boxes.append(drive_box)
            drive_widgets.append(drive)

        accordion = Accordion(children=drive_boxes,
                              selected_index=len(drive_boxes) - 1)
        for idx, this_title in enumerate(drive_titles):
            accordion.set_title(idx, this_title)
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
        elif plot_type['new'] == 'spectogram':
            freqs = np.arange(20., 100., 1.)
            variables['dpls'][0].plot_tfr_morlet(freqs)


def on_button_clicked(log_out, plot_out, drive_widgets, variables, b):
    """Run the simulation and plot outputs."""
    for drive in drive_widgets:
        if drive['type'] == 'Poisson':
            weights_ampa, weights_nmda, rate_constant = dict(), dict(), dict()
            for k, v in drive['rate_constant'].items():
                if v.value > 0:
                    weights_ampa[k] = drive['weights_ampa'][k].value
                    weights_nmda[k] = drive['weights_nmda'][k].value
                    rate_constant[k] = drive['rate_constant'][k].value
            variables['net'].add_poisson_drive(
                name=drive['name'],
                tstart=drive['tstart'].value,
                tstop=drive['tstop'].value,
                rate_constant=rate_constant,
                location=drive['location'].value,
                weights_ampa=weights_ampa,
                weights_nmda=weights_nmda,
                space_constant=100.0
            )
        elif drive['type'] == 'Evoked':
            variables['net'].add_evoked_drive(
                name=drive['name'],
                mu=drive['mu'].value,
                sigma=drive['sigma'].value,
                numspikes=drive['numspikes'].value,
                sync_within_trial=False,
                location=drive['location'].value,
                weights_ampa=dict(L5_pyramidal=drive['weights_ampa_L5Pyr'].value),
                weights_nmda=dict(L5_pyramidal=drive['weights_nmda_L5Pyr'].value),
                space_constant=3.0
            )
        elif drive['type'] == 'Rhythmic':
            variables['net'].add_bursty_drive(
                name=drive['name'],
                tstart=drive['tstart'].value,
                tstart_std=drive['tstart_std'].value,
                burst_rate=drive['burst_rate'].value,
                burst_std=drive['burst_std'].value,
                location=drive['location'].value
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

    variables = dict(net=None, dpls=None)
    variables['net'] = Network(params, add_drives_from_params=False)

    # Output windows
    log_out = Output(layout={'border': '1px solid gray', 'height': '150px',
                             'overflow_y': 'auto'})
    plot_out = Output(layout={'border': '1px solid gray', 'height': '350px'})

    # header_button
    header_button = create_expanded_button('HUMAN NEOCORTICAL NEUROSOLVER',
                                           'success', height='40px')

    # Sliders to change local-connectivity Params
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

    # accordians
    boxes = [VBox(slider) for slider in sliders]
    titles = ['Layer 2/3 Pyr', 'Layer 5 Pyr', 'Layer 2 Bas', 'Layer 5 Bas']
    accordian = Accordion(children=boxes)
    for idx, title in enumerate(titles):
        accordian.set_title(idx, title)

    # Dropdown for different drives
    layout = Layout(width='200px', height='auto')
    drives_dropdown = Dropdown(
        options=['Evoked', 'Poisson', 'Rhythmic', ''],
        value='',
        description='Drive:',
        disabled=False,
        layout=layout
    )

    # XXX: should be simpler to use Stacked class starting
    # from IPywidgets > 8.0
    interactive(add_drive_widget, drive_type='Evoked')
    drives_dropdown.observe(add_drive_widget, 'value')
    drives_options = VBox([drives_dropdown, drives_out])

    # Tabs for left pane
    left_tab = Tab()
    left_tab.children = [accordian, drives_options]
    titles = ['Cell connectivity', 'Drives']
    for idx, title in enumerate(titles):
        left_tab.set_title(idx, title)

    # Dropdown menu to switch between plots
    def _update_plot_window(plot_type):
        return update_plot_window(variables, plot_out, plot_type)

    dropdown = Dropdown(
        options=['input histogram', 'current dipole', 'spikes', 'spectogram'],
        value='current dipole',
        description='Plot:',
        disabled=False,
    )
    interactive(_update_plot_window, plot_type='current dipole')
    dropdown.observe(_update_plot_window, 'value')

    # Run and load button
    run_button = create_expanded_button('Run', 'success', height='30px')
    style = {'button_color': '#8A2BE2', 'font_color': 'white'}
    load_button = FileUpload(accept='.json', multiple=False, style=style,
                             description='Load network', button_style='success')

    def _on_upload_change(change):
        import json
        import codecs

        if len(change['owner'].value) == 0:
            return

        file_uploaded = change['owner'].value
        json_data = list(file_uploaded.values())[0]['content']
        params = json.loads(json_data)

    def _on_button_clicked(b):
        return on_button_clicked(log_out, plot_out, drive_widgets, variables,
                                 b)

    load_button.observe(_on_upload_change)
    run_button.on_click(_on_button_clicked)
    footer = HBox([run_button, load_button, dropdown])

    right_sidebar = VBox([plot_out, log_out])

    # Final layout of the app
    hnn_gui = AppLayout(header=header_button,
                        left_sidebar=left_tab,
                        right_sidebar=right_sidebar,
                        footer=footer,
                        pane_widths=['380px', 1, 1],
                        pane_heights=[1, '500px', 1])
    return hnn_gui
