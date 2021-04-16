"""IPywidgets GUI."""
# Authors: Mainak Jas <mjas@mgh.harvard.edu>

import os.path as op
from functools import partial, update_wrapper

import hnn_core
from hnn_core import simulate_dipole, read_params, Network

from IPython.display import display

from ipywidgets import (FloatSlider, Dropdown, Accordion, Tab,
                        interactive_output, HBox, VBox,
                        FloatText, interactive, Dropdown, interact, Output,
                        RadioButtons, Button)
from ipywidgets import Layout, AppLayout, fixed


drives_out = Output()
drive_widgets = list()
drive_titles = list()
drive_boxes = list()


def create_expanded_button(description, button_style):
    return Button(description=description, button_style=button_style,
                  layout=Layout(height='10', width='auto'))


def update_params(params, **updates):
    params.update(dict(**updates))
    return params


def _get_min(v):
    if v < 0:
        return v * 10
    else:
        return v * 0.1


def _get_max(v):
    if v > 0:
        return v * 10
    else:
        return v * 0.1


def _get_sliders(params, param_keys):
    """Get sliders"""
    style = {'description_width': 'initial'}
    sliders = list()
    for d in param_keys:
        min_val = _get_min(params[d])
        max_val = _get_max(params[d])
        step = (max_val - min_val) / 10.
        slider = FloatSlider(
            value=params[d], min=min_val, max=max_val, step=step,
            description=d,
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.2e',
            style=style)
        sliders.append(slider)

    _update_params = partial(update_params, params)
    interactive_output(update_params, {s.description: s for s in sliders})
    return sliders


def add_drive_widget(drive_type):

    layout = Layout(width='200px', height='auto')
    drives_out.clear_output()
    with drives_out:
        drive_title = drive_type['new'] + str(len(drive_boxes))

        if drive_type['new'] == 'Rhythmic':
            tstart = FloatText(value=7.5, description='Start time:', layout=layout)
            tstart_std = FloatText(value=7.5, description='Start time dev:', layout=layout)
            tstop = FloatText(value=7.5, description='Stop time:', layout=layout)
            burst_rate = FloatText(value=7.5, description='Burst rate:', layout=layout)
            burst_std = FloatText(value=7.5, description='Burst std dev:', layout=layout)
            location = RadioButtons(options=['proximal', 'distal'])

            drive_box = VBox([tstart, tstart_std, tstop, burst_rate, burst_std, location])
            drive = dict(type='Rhythmic', name=drive_title,
                         tstart=tstart,
                         tstart_std=tstart_std,
                         burst_rate=burst_rate,
                         burst_std=burst_std,
                         location=location
            )
        elif drive_type['new'] == 'Poisson':            
            tstart = FloatText(value=7.5, description='Start time:', layout=layout)
            tstop = FloatText(value=8.5, description='Stop time:', layout=layout)
            rate_constant = FloatText(value=8.5, description='Rate constant:', layout=layout)
            location = RadioButtons(options=['proximal', 'distal'])
            weights_ampa_L5Pyr = FloatText(value=8.5, description='AMPA (L5 Pyr):', layout=layout)
            weights_nmda_L5Pyr = FloatText(value=8.5, description='NMDA (L5 Pyr):', layout=layout)
            
            drive_box = VBox([tstart, tstop, rate_constant, location,
                              weights_ampa_L5Pyr, weights_nmda_L5Pyr])
            drive = dict(type='Poisson', name=drive_title,
                         tstart=tstart, 
                         tstop=tstop,
                         rate_constant=rate_constant, 
                         location=location, 
                         weights_ampa_L5Pyr=weights_ampa_L5Pyr, 
                         weights_nmda_L5Pyr=weights_nmda_L5Pyr
                        )
        elif drive_type['new'] == 'Evoked':            
            mu = FloatText(value=7.5, description='Mean time:', layout=layout)
            sigma = FloatText(value=8.5, description='Std dev time:', layout=layout)
            numspikes = FloatText(value=8.5, description='Number of spikes:', layout=layout)
            location = RadioButtons(options=['proximal', 'distal'])
            weights_ampa_L5Pyr = FloatText(value=8.5, description='AMPA (L5 Pyr):', layout=layout)
            weights_nmda_L5Pyr = FloatText(value=8.5, description='NMDA (L5 Pyr):', layout=layout)

            drive_box = VBox([mu, sigma, numspikes, location,
                              weights_ampa_L5Pyr, weights_nmda_L5Pyr])
            drive = dict(type='Evoked', name=drive_title,
                         mu=mu,
                         sigma=sigma, 
                         numspikes=numspikes, 
                         sync_within_trial=False, 
                         location=location, 
                         weights_ampa_L5Pyr=weights_ampa_L5Pyr, 
                         weights_nmda_L5Pyr=weights_nmda_L5Pyr, 
                         space_constant=3.0)

        drive_titles.append(drive_title)
        drive_boxes.append(drive_box)
        drive_widgets.append(drive)

        accordion = Accordion(children=drive_boxes,
                              selected_index=len(drive_boxes) - 1)
        for idx, this_title in enumerate(drive_titles):
            accordion.set_title(idx, this_title)
        display(accordion)


def update_plot(variables, plot_out, plot_type):
    plot_out.clear_output()

    if not (plot_type['type'] == 'change' and plot_type['name'] == 'value'):
        return

    with plot_out:
        if plot_type['new'] == 'spikes':
            variables['net'].cell_response.plot_spikes_raster()
        elif plot_type['new'] == 'dipole current':
            variables['dpls'][0].plot()
        elif plot_type['new'] == 'input histogram':
            variables['net'].cell_response.plot_spikes_hist()


def on_button_clicked(log_out, plot_out, drive_widgets, variables, b):
    """Run the simulation and plot outputs."""
    with log_out:
        for drive in drive_widgets:
            if drive['type'] == 'Poisson':
                variables['net'].add_poisson_drive(
                    name=drive['name'],
                    tstart=drive['tstart'].value, 
                    tstop=drive['tstop'].value,
                    rate_constant=dict(L5_pyramidal=drive['rate_constant'].value), 
                    location=drive['location'].value,
                    weights_ampa=dict(L5_pyramidal=drive['weights_ampa_L5Pyr'].value), 
                    weights_nmda=dict(L5_pyramidal=drive['weights_nmda_L5Pyr'].value), 
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
        variables['dpls'] = simulate_dipole(variables['net'], n_trials=1)
    with plot_out:
        variables['dpls'][0].plot()


def run_hnn_gui():

    hnn_core_root = op.join(op.dirname(hnn_core.__file__))

    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    variables = dict(net=None, dpls=None)
    variables['net'] = Network(params, add_drives_from_params=False)

    # Output windows
    log_out = Output(layout={'border': '1px solid gray'})
    plot_out = Output(layout={'border': '1px solid gray'})

    # header_button
    header_button = create_expanded_button('Human Neocortical Neurosolver',
                                           'success')

    # Sliders to change local-connectivity Params
    sliders = [_get_sliders(params, ['gbar_L2Pyr_L2Pyr_ampa', 'gbar_L2Pyr_L2Pyr_nmda',
                                     'gbar_L2Basket_L2Pyr_gabaa', 'gbar_L2Basket_L2Pyr_gabab']),
               _get_sliders(params, ['gbar_L2Pyr_L5Pyr', 'gbar_L2Basket_L5Pyr', 'gbar_L5Pyr_L5Pyr_ampa',
                                     'gbar_L5Pyr_L5Pyr_nmda', 'gbar_L5Basket_L5Pyr_gabaa',
                                     'gbar_L5Basket_L5Pyr_gabab']),
               _get_sliders(params, ['gbar_L2Pyr_L2Basket', 'gbar_L2Basket_L2Basket']),
               _get_sliders(params, ['gbar_L2Pyr_L5Pyr', 'gbar_L2Basket_L5Pyr'])]

    # accordians
    boxes = [VBox(slider) for slider in sliders]
    titles = ['Layer 2/3 Pyr', 'Layer 5 Pyr', 'Layer 2 Bas', 'Layer 5 Bas']
    accordian = Accordion(children=boxes)
    for idx, title in enumerate(titles):
        accordian.set_title(idx, title)

    # Dropdown for different drives
    layout = Layout(width='200px', height='auto')
    drives_dropdown = Dropdown(
        options=['Evoked', 'Poisson', 'Rhythmic'],
        value='Evoked',
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
    dropdown = Dropdown(
        options=['input histogram', 'dipole current', 'spikes'],
        value='dipole current',
        description='Plot:',
        disabled=False,
    )

    interactive(update_plot, plot_type='dipole current',
                variables=fixed(variables), plot_out=fixed(plot_out))
    dropdown.observe(lambda plot_type: update_plot(variables, plot_out, plot_type), 'value')

    # Run button
    run_button = create_expanded_button('Run', 'success')
    load_button = create_expanded_button('Load parameters', 'success')

    run_button.on_click(lambda b: on_button_clicked(log_out, plot_out, drive_widgets, variables, b))

    footer = HBox([run_button, load_button, dropdown])

    # Final layout of the app
    hnn_gui = AppLayout(header=header_button,
                        left_sidebar=left_tab,
                        center=log_out,
                        right_sidebar=plot_out,
                        footer=footer,
                        pane_widths=['380px', 1, 1],
                        pane_heights=[1, '500px', 1])
    return hnn_gui
