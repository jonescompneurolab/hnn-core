"""External drives to network."""

# Authors: Christopher Bailey <bailey.cj@gmail.com>

from .params import (_extract_bias_specs_from_hnn_params,
                     _extract_drive_specs_from_hnn_params)


def _get_target_populations(weights_ampa, weights_nmda):
    # allow passing weights as None, but make iterable here
    if weights_ampa is None:
        weights_ampa = dict()
    if weights_nmda is None:
        weights_nmda = dict()
    target_populations = (set(weights_ampa.keys()) | set(weights_nmda.keys()))
    return target_populations, weights_ampa, weights_nmda


def _check_drive_parameter_values(drive_type, **kwargs):
    if 'sigma' in kwargs:
        if kwargs['sigma'] < 0.:
            raise ValueError('Standard deviation cannot be negative')
    if 'numspikes' in kwargs:
        if not kwargs['numspikes'] > 0:
            raise ValueError('Number of spikes must be greater than zero')
    if 'tstart' in kwargs:
        if kwargs['tstart'] < 0:
            raise ValueError(f'Start time of {drive_type} drive cannot be '
                             'negative')
    if 'tstop' in kwargs:
        if kwargs['tstop'] < 0.:
            raise ValueError(f'End time of {drive_type} drive cannot be '
                             'negative')
    if 'tstop' in kwargs and 'sim_end_time' in kwargs:
        if kwargs['tstop'] > kwargs['sim_end_time']:
            raise ValueError(f"End time of {drive_type} drive cannot exceed "
                             f"simulation end time {kwargs['sim_end_time']}. "
                             f"Got {kwargs['tstop']}.")
    if 'tstart' in kwargs and 'tstop' in kwargs:
        if kwargs['tstop'] - kwargs['tstart'] < 0.:
            raise ValueError(f'Duration of {drive_type} drive cannot be '
                             'negative')

    if ('numspikes' in kwargs and 'spike_isi' in kwargs and
            'burst_rate' in kwargs):
        nspikes = kwargs['numspikes']
        isi = kwargs['spike_isi']
        burst_period = 1000. / kwargs['burst_rate']
        if (nspikes - 1) * isi > burst_period:
            raise ValueError('Burst duration cannot be greater than period')


def _check_poisson_rates(rate_constant, target_populations, all_cell_types):
    if isinstance(rate_constant, dict):
        constants_provided = set(rate_constant.keys())
        if not target_populations.issubset(constants_provided):
            raise ValueError(
                f"Rate constants not provided for all target cell "
                f"populations ({target_populations})")
        if not constants_provided.issubset(all_cell_types):
            offending_keys = constants_provided.difference(all_cell_types)
            raise ValueError(
                f"Rate constant provided for unknown target cell "
                f"population: {offending_keys}")
    else:
        rate_constant = {key: rate_constant for key in all_cell_types}
    for key, val in rate_constant.items():
        if not val > 0.:
            raise ValueError(
                f"Rate constant must be positive ({key}, {val})")


def _add_drives_from_params(net):
    drive_specs = _extract_drive_specs_from_hnn_params(
        net.params, net.cellname_list)
    bias_specs = _extract_bias_specs_from_hnn_params(
        net.params, net.cellname_list)

    for drive_name in sorted(drive_specs.keys()):  # order matters
        specs = drive_specs[drive_name]
        if specs['type'] == 'evoked':
            net.add_evoked_drive(
                drive_name, mu=specs['dynamics']['mu'],
                sigma=specs['dynamics']['sigma'],
                numspikes=specs['dynamics']['numspikes'],
                sync_within_trial=specs['dynamics']
                                        ['sync_within_trial'],
                weights_ampa=specs['weights_ampa'],
                weights_nmda=specs['weights_nmda'],
                location=specs['location'], seedcore=specs['seedcore'],
                synaptic_delays=specs['synaptic_delays'],
                space_constant=specs['space_constant'])
        elif specs['type'] == 'poisson':
            net.add_poisson_drive(
                drive_name, tstart=specs['dynamics']['tstart'],
                tstop=specs['dynamics']['tstop'],
                rate_constant=specs['dynamics']['rate_constant'],
                weights_ampa=specs['weights_ampa'],
                weights_nmda=specs['weights_nmda'],
                location=specs['location'], seedcore=specs['seedcore'],
                synaptic_delays=specs['synaptic_delays'],
                space_constant=specs['space_constant'])
        elif specs['type'] == 'gaussian':
            net.add_evoked_drive(  # 'gaussian' is just evoked
                drive_name, mu=specs['dynamics']['mu'],
                sigma=specs['dynamics']['sigma'],
                numspikes=specs['dynamics']['numspikes'],
                weights_ampa=specs['weights_ampa'],
                weights_nmda=specs['weights_nmda'],
                location=specs['location'], seedcore=specs['seedcore'],
                synaptic_delays=specs['synaptic_delays'],
                space_constant=specs['space_constant'])
        elif specs['type'] == 'bursty':
            net.add_bursty_drive(
                drive_name,
                distribution=specs['dynamics']['distribution'],
                tstart=specs['dynamics']['tstart'],
                tstart_std=specs['dynamics']['tstart_std'],
                tstop=specs['dynamics']['tstop'],
                burst_rate=specs['dynamics']['burst_rate'],
                burst_std=specs['dynamics']['burst_std'],
                numspikes=specs['dynamics']['numspikes'],
                spike_isi=specs['dynamics']['spike_isi'],
                repeats=specs['dynamics']['repeats'],
                weights_ampa=specs['weights_ampa'],
                weights_nmda=specs['weights_nmda'],
                location=specs['location'],
                space_constant=specs['space_constant'],
                synaptic_delays=specs['synaptic_delays'],
                seedcore=specs['seedcore'])

    # add tonic biases if present in params
    for cellname in bias_specs['tonic']:
        net.add_tonic_bias(
            cell_type=cellname,
            amplitude=bias_specs['tonic'][cellname]['amplitude'],
            t0=bias_specs['tonic'][cellname]['t0'],
            T=bias_specs['tonic'][cellname]['T'])

    net._instantiate_drives(n_trials=net.params['N_trials'])
