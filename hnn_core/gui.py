"""IPywidgets GUI."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Huzi Cheng <hzcheng15@icloud.com>

import codecs
import multiprocessing
import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (HTML, Accordion, AppLayout, BoundedIntText,
                        BoundedFloatText, Button, Dropdown, FileUpload,
                        FloatLogSlider, FloatText, GridspecLayout, HBox,
                        IntText, Layout, Output, RadioButtons, Tab, Text, VBox,
                        interactive_output)

import hnn_core
from hnn_core import (JoblibBackend, MPIBackend, jones_2009_model, read_params,
                      simulate_dipole)
from hnn_core.params import (_extract_drive_specs_from_hnn_params, _read_json,
                             _read_legacy_params)
from hnn_core.viz import plot_dipole


def create_expanded_button(description, button_style, height, disabled=False):
    style = {'button_color': '#8A2BE2'}
    return Button(description=description,
                  button_style=button_style,
                  layout=Layout(height=height, width='auto'),
                  style=style,
                  disabled=disabled)


def _get_sliders(params, param_keys):
    """Get sliders"""
    style = {'description_width': '150px'}
    sliders = list()
    for d in param_keys:
        slider = FloatLogSlider(value=params[d],
                                min=-5,
                                max=1,
                                step=0.2,
                                description=d.split('gbar_')[1],
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='.2e',
                                style=style)
        sliders.append(slider)

    def _update_params(variables, **updates):
        params.update(dict(**updates))

    interactive_output(_update_params, {s.description: s for s in sliders})
    return sliders


def _get_cell_specific_widgets(
    layout,
    style,
    location,
    data=None,
):
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
            if k in data:
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


def _get_rhythmic_widget(
    name,
    tstop_widget,
    layout,
    style,
    location,
    data=None,
    default_weights_ampa=None,
    default_weights_nmda=None,
    default_delays=None,
):
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
        value=tstop_widget.value
        if default_data['tstop'] == 0 else default_data['tstop'],
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


def _get_poisson_widget(
    name,
    tstop_widget,
    layout,
    style,
    location,
    data=None,
    default_weights_ampa=None,
    default_weights_nmda=None,
    default_delays=None,
):
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
        value=tstop_widget.value
        if default_data['tstop'] == 0 else default_data['tstop'],
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


def _get_evoked_widget(
    name,
    layout,
    style,
    location,
    data=None,
    default_weights_ampa=None,
    default_weights_nmda=None,
    default_delays=None,
):
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

        if drive_type == 'Rhythmic':
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
        elif drive_type == 'Evoked':
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
                'Evoked',
                'Poisson',
                'Rhythmic',
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
                accordion.set_title(idx, drive['name'])
            display(accordion)


def _debug_update_plot_window(variables, _plot_out, plot_type, idx):
    update_plot_window(variables, _plot_out, plot_type)


def update_plot_window(variables, _plot_out, plot_type):
    _plot_out.clear_output()
    if not (plot_type['type'] == 'change' and plot_type['name'] == 'value'):
        return

    with _plot_out:
        if plot_type['new'] == 'spikes':
            fig, ax = plt.subplots()
            variables['net'].cell_response.plot_spikes_raster(ax=ax)

        elif plot_type['new'] == 'current dipole':
            fig, ax = plt.subplots()
            # variables['dpls'][0].plot(ax=ax)
            plot_dipole(variables['dpls'], ax=ax, average=True)

        elif plot_type['new'] == 'input histogram':
            # BUG: got error here, need a better way to handle exception
            fig, ax = plt.subplots()
            variables['net'].cell_response.plot_spikes_hist(ax=ax)

        elif plot_type['new'] == 'PSD':
            fig, ax = plt.subplots()
            variables['dpls'][0].plot_psd(fmin=0, fmax=50, ax=ax)

        elif plot_type['new'] == 'spectogram':
            freqs = np.arange(10., 100., 1.)
            n_cycles = freqs / 8.
            fig, ax = plt.subplots()
            variables['dpls'][0].plot_tfr_morlet(freqs,
                                                 n_cycles=n_cycles,
                                                 ax=ax)
        elif plot_type['new'] == 'network':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            variables['net'].plot_cells(ax=ax)


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
            print(f"""load drive:
                (name={drive_name},
                type={specs['type']},
                seed={specs['event_seed']},
                space_constant={specs['space_constant']}
                )
                """)
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


def on_upload_change(
    change,
    sliders,
    params,
    tstop,
    tstep,
    log_out,
    variables,
    # for adding drives
    drive_boxes,
    drive_widgets,
    drives_out,
):
    if len(change['owner'].value) == 0:
        return

    params_fname = change['owner'].metadata[0]['name']
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
                key = 'gbar_' + sl.description
                sl.value = params_network[key]

        if 'tstop' in params_network.keys():
            tstop.value = params_network['tstop']
        if 'dt' in params_network.keys():
            tstep.value = params_network['dt']

        params.update(params_network)
    load_drives(variables, params, log_out, drives_out, drive_widgets,
                drive_boxes, tstop)


def run_button_clicked(log_out, drive_widgets, variables, tstep, tstop,
                       ntrials, backend_selection, mpi_cmd, joblib_cores,
                       params, plot_outputs_list, plot_dropdowns_list, b):
    """Run the simulation and plot outputs."""
    log_out.clear_output()
    with log_out:
        print(f"drive_widgets length={len(drive_widgets)}")
        params['dt'] = tstep.value
        params['tstop'] = tstop.value
        variables['net'] = jones_2009_model(
            params,
            add_drives_from_params=False,
        )
        # add drives to network
        for drive in drive_widgets:
            print(drive['type'], drive['name'], drive['seedcore'].value)
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
            elif drive['type'] == 'Evoked':
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
            elif drive['type'] == 'Rhythmic':
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

        print("start simulation")
        if backend_selection.value == "MPI":
            variables['backend'] = MPIBackend(
                n_procs=multiprocessing.cpu_count() - 1, mpi_cmd=mpi_cmd.value)
        else:
            variables['backend'] = JoblibBackend(n_jobs=joblib_cores.value)
            print(f"Using Joblib with {joblib_cores.value} core(s).")
        with variables['backend']:
            variables['dpls'] = simulate_dipole(variables['net'],
                                                tstop=tstop.value,
                                                n_trials=ntrials.value)

            window_len, scaling_factor = 30, 3000
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


def handle_backend_change(backend_type, backend_config, mpi_cmd, joblib_cores):
    backend_config.clear_output()
    with backend_config:
        if backend_type == "MPI":
            display(mpi_cmd)
        elif backend_type == "Joblib":
            display(joblib_cores)


def init_left_right_viz_layout(plot_outputs,
                               plot_dropdowns,
                               window_height,
                               variables,
                               plot_options,
                               border='1px solid gray'):
    height_plot = window_height
    plot_outputs_L = Output(layout={'border': border, 'height': height_plot})

    plot_dropdown_L = Dropdown(
        options=plot_options,
        value=plot_options[0],
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
        value=plot_options[1],
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

    grid = GridspecLayout(1, 2, height=window_height)
    grid[0, 0] = VBox([plot_dropdown_L, plot_outputs_L])
    grid[0, 1] = VBox([plot_dropdown_R, plot_outputs_R])
    return grid


def init_upper_down_viz_layout(plot_outputs,
                               plot_dropdowns,
                               window_height,
                               variables,
                               plot_options,
                               border='1px solid gray'):
    height_plot = window_height
    plot_outputs_U = Output(layout={
        'border': border,
        'height': f"{float(height_plot[:-2])/2}px"
    })

    plot_dropdown_U = Dropdown(
        options=plot_options,
        value=plot_options[0],
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
        value=plot_options[1],
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
                          layout_option="L-R"):
    plot_options = [
        'current dipole', 'input histogram', 'spikes', 'PSD', 'spectogram',
        'network'
    ]
    viz_window.clear_output()
    while len(plot_outputs) > 0:
        plot_outputs.pop()
        plot_dropdowns.pop()

    with viz_window:
        # Left-Rright configuration
        if layout_option == "L-R":
            grid = init_left_right_viz_layout(plot_outputs, plot_dropdowns,
                                              window_height, variables,
                                              plot_options)
        # Upper-Down configuration
        elif layout_option == "U-D":
            grid = init_upper_down_viz_layout(plot_outputs, plot_dropdowns,
                                              window_height, variables,
                                              plot_options)
        # TODO: 2x2

        display(grid)


def run_hnn_gui():
    """Create the HNN GUI."""

    hnn_core_root = op.join(op.dirname(hnn_core.__file__))

    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    drive_widgets = list()
    drive_boxes = list()
    variables = dict(net=None, dpls=None)

    plot_outputs_list = list()
    plot_dropdowns_list = list()

    def _run_button_clicked(b):
        return run_button_clicked(log_out, drive_widgets, variables, tstep,
                                  tstop, ntrials, backend_selection, mpi_cmd,
                                  joblib_cores, params, plot_outputs_list,
                                  plot_dropdowns_list, b)

    def _on_upload_change(change):
        return on_upload_change(change, sliders, params, tstop, tstep, log_out,
                                variables, drive_boxes, drive_widgets,
                                drives_out)
        # BUG: capture does not work, use log_out explicitly
        # return on_upload_change(change, sliders, params)

    def _delete_drives_clicked(b):
        drives_out.clear_output()
        # black magic: the following does not work
        # global drive_widgets; drive_widgets = list()
        while len(drive_widgets) > 0:
            drive_widgets.pop()
            drive_boxes.pop()

    # Output windows
    drives_out = Output()  # window to add new drives

    log_out_heiht = "100px"
    log_out = Output(layout={
        'border': '1px solid gray',
        'height': log_out_heiht,
        'overflow_y': 'auto'
    })
    viz_width = "1000px"
    viz_height = "500px"
    viz_window = Output(layout={
        'height': viz_height,
        'width': viz_width,
    })

    # header_button
    header_button = create_expanded_button('HUMAN NEOCORTICAL NEUROSOLVER',
                                           'success',
                                           height='40px',
                                           disabled=True)

    # Simulation parameters
    tstop = FloatText(value=170, description='tstop (ms):', disabled=False)
    tstep = FloatText(value=0.025, description='tstep (ms):', disabled=False)
    ntrials = IntText(value=1, description='Trials:', disabled=False)
    # temporarily keep this

    viz_layout_selection = Dropdown(
        options=[('Horizontal', 'L-R'), ('Vertical', 'U-D')],
        value='L-R',
        description='Layout:',
    )
    # initialize
    initialize_viz_window(
        viz_window,
        variables,
        plot_outputs_list,
        plot_dropdowns_list,
        viz_width,
        viz_height,
        layout_option=viz_layout_selection.value,
    )

    def handle_viz_layout_change(layout_option):
        return initialize_viz_window(
            viz_window,
            variables,
            plot_outputs_list,
            plot_dropdowns_list,
            viz_width,
            viz_height,
            layout_option=layout_option.new,
        )

    viz_layout_selection.observe(handle_viz_layout_change, 'value')

    backend_selection = Dropdown(
        options=[('Joblib', 'Joblib'), ('MPI', 'MPI')],
        value='MPI' if os.getenv("USEMPI", '0') == '1' else 'Joblib',
        description='Backend:',
    )

    mpi_cmd = Text(value='mpiexec',
                   placeholder='Fill if applies',
                   description='MPI cmd:',
                   disabled=False)

    joblib_cores = BoundedIntText(value=1,
                                  min=1,
                                  max=multiprocessing.cpu_count(),
                                  description='Cores:',
                                  disabled=False)

    backend_config = Output()

    def _handle_backend_change(backend_type):
        return handle_backend_change(backend_type.new, backend_config, mpi_cmd,
                                     joblib_cores)

    handle_backend_change(backend_selection.value, backend_config, mpi_cmd,
                          joblib_cores)
    backend_selection.observe(_handle_backend_change, 'value')

    simulation_box = VBox([
        tstop,
        tstep,
        ntrials,
        backend_selection,
        backend_config,
    ])

    # Sliders to change local-connectivity params
    sliders = [
        _get_sliders(params, [
            'gbar_L2Pyr_L2Pyr_ampa', 'gbar_L2Pyr_L2Pyr_nmda',
            'gbar_L2Basket_L2Pyr_gabaa', 'gbar_L2Basket_L2Pyr_gabab'
        ]),
        _get_sliders(params, [
            'gbar_L2Pyr_L5Pyr', 'gbar_L2Basket_L5Pyr', 'gbar_L5Pyr_L5Pyr_ampa',
            'gbar_L5Pyr_L5Pyr_nmda', 'gbar_L5Basket_L5Pyr_gabaa',
            'gbar_L5Basket_L5Pyr_gabab'
        ]),
        _get_sliders(params,
                     ['gbar_L2Pyr_L2Basket', 'gbar_L2Basket_L2Basket']),
        _get_sliders(params, [
            'gbar_L2Pyr_L5Basket', 'gbar_L5Pyr_L5Basket',
            'gbar_L5Basket_L5Basket'
        ])
    ]

    # accordians to group local-connectivity by cell type
    boxes = [VBox(slider) for slider in sliders]
    titles = [
        'Layer 2/3 Pyramidal', 'Layer 5 Pyramidal', 'Layer 2 Basket',
        'Layer 5 Basket'
    ]
    cell_connectivity = Accordion(children=boxes)
    for idx, title in enumerate(titles):
        cell_connectivity.set_title(idx, title)

    # Dropdown for different drives
    layout = Layout(width='200px', height='100px')

    drive_type_selection = RadioButtons(
        options=['Evoked', 'Poisson', 'Rhythmic'],
        value='Evoked',
        description='Drive:',
        disabled=False,
        layout=layout)

    location_selection = RadioButtons(options=['proximal', 'distal'],
                                      value='proximal',
                                      description='Location',
                                      disabled=False,
                                      layout=layout)

    add_drive_button = create_expanded_button('Add drive',
                                              'primary',
                                              height='30px')

    def _add_drive_button_clicked(b):
        return add_drive_widget(drive_type_selection.value, drive_boxes,
                                drive_widgets, drives_out, tstop,
                                location_selection.value)

    add_drive_button.on_click(_add_drive_button_clicked)
    drive_selections = VBox(
        [HBox([drive_type_selection, location_selection]), add_drive_button])

    # XXX: should be simpler to use Stacked class starting
    # from IPywidgets > 8.0
    drives_options = VBox([drive_selections, drives_out])

    # Tabs for left pane
    left_tab = Tab()
    left_tab.children = [simulation_box, cell_connectivity, drives_options]
    titles = ['Simulation', 'Cell connectivity', 'Drives']
    for idx, title in enumerate(titles):
        left_tab.set_title(idx, title)

    # Run, delete drives and load button
    run_button = create_expanded_button('Run', 'success', height='30px')

    style = {'button_color': '#8A2BE2', 'font_color': 'white'}
    load_button = FileUpload(accept='.json,.param',
                             multiple=False,
                             style=style,
                             description='Load network',
                             button_style='success')
    delete_button = create_expanded_button('Delete drives',
                                           'success',
                                           height='30px')

    load_button.observe(_on_upload_change)
    run_button.on_click(_run_button_clicked)

    delete_button.on_click(_delete_drives_clicked)
    left_width = '380px'
    footer = VBox([
        HBox([
            HBox([run_button, load_button, delete_button],
                 layout={"width": left_width}),
            viz_layout_selection,
        ]), log_out
    ])

    # initialize drive ipywidgets
    load_drives(variables, params, log_out, drives_out, drive_widgets,
                drive_boxes, tstop)

    # Final layout of the app
    hnn_gui = AppLayout(
        header=header_button,
        left_sidebar=left_tab,
        right_sidebar=viz_window,
        footer=footer,
        pane_widths=[left_width, '0px', viz_width],
        pane_heights=['50px', viz_height, "1"],
    )
    return hnn_gui


def launch():
    from voila.app import main
    notebook_path = op.join(op.dirname(__file__), '..', 'hnn_widget.ipynb')
    main([notebook_path])
