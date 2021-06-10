"""External drives to network."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import numpy as np

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
        n_spikes = kwargs['numspikes']
        isi = kwargs['spike_isi']
        burst_period = 1000. / kwargs['burst_rate']
        burst_duration = (n_spikes - 1) * isi
        if burst_duration > burst_period:
            raise ValueError(f'Burst duration ({burst_duration}s) cannot'
                             f' be greater than burst period ({burst_period}s)'
                             'Consider increasing the spike ISI or burst rate')


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
        net._params, list(net.cell_types.keys()))
    bias_specs = _extract_bias_specs_from_hnn_params(
        net._params, list(net.cell_types.keys()))

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
            tstop=bias_specs['tonic'][cellname]['tstop'])

    net._instantiate_drives(n_trials=net._params['N_trials'])


def _get_prng(seed, gid, sync_evinput=False):
    """Random generator for this instance.

    Parameters
    ----------
    seed : int
        The seed for random state generator.
    gid : int
        The cell ID
    sync_evinput : bool
        If True, all cells get the same prng

    Returns
    -------
    prng : instance of RandomState
        The seed for events assuming a given start time.
    prng2 : instance of RandomState
        The seed for generating randomized start times.
        Used in _create_bursty_input
    """
    # XXX: some param files use seed < 0 but numpy
    # does not allow this.
    if seed >= 0:
        # only used for randomisation of t0 of bursty drives
        prng2 = np.random.RandomState(seed)
    else:
        prng2 = None

    if not sync_evinput:
        seed = seed + gid

    prng = np.random.RandomState(seed)
    return prng, prng2


def _drive_cell_event_times(drive_type, drive_conn, dynamics,
                            trial_idx=0, drive_cell_gid=0, seedcore=0):
    """Generate event times for one artificial drive cell based on dynamics.

    Parameters
    ----------
    drive_type : str
        The drive type, which is one of
        'poisson' : Poisson-distributed dynamics from t0 to T
        'gaussian' : Gaussian-distributed dynamics from t0 to T
        'evoked' : Spikes occur at specified time (mu) with dispersion (sigma)
        'bursty' : As opposed to other drive types, these have timing that is
        identical (synchronous) for all real cells in the network.
    drive_conn : dict
        A drive is associated with a number of 'artificial cells', each
        with its spatial connectivity (and temporal dynamics). drive_conn
        defines AMPA and NMDA weights, and the cell target (e.g. 'L2_basket')
    dynamics : dict
        Parameters of the event time dynamics to simulate
    trial_idx : int
        The index number of the current trial of a simulation (default=1).
    drive_cell_gid : int
        Optional gid of current artificial cell (used for seeding)
    seedcore : int
        Optional initial seed for random number generator.

    Returns
    -------
    event_times : list
        The event times at which spikes occur.
    """
    sync_evinput = False
    if 'sync_within_trial' in dynamics:
        sync_evinput = dynamics['sync_within_trial']

    prng, prng2 = _get_prng(seed=seedcore + trial_idx,
                            gid=drive_cell_gid,
                            sync_evinput=sync_evinput)

    # check drive name validity, allowing substring matches
    valid_drives = ['evoked', 'poisson', 'gaussian', 'bursty']
    # NB check if drive_type has a valid substring, not vice versa
    matches = [f for f in valid_drives if f in drive_type]
    if len(matches) == 0:
        raise ValueError('Invalid external drive: %s' % drive_type)
    elif len(matches) > 1:
        raise ValueError('Ambiguous external drive: %s' % drive_type)

    # Return values not checked: False if all weights for given drive type
    # are zero. Designed to be silent so that zeroing input weights
    # effectively disables each.
    n_ampa_nmda_weights = (len(drive_conn['ampa'].keys()) +
                           len(drive_conn['nmda'].keys()))
    target_syn_weights_zero = True if n_ampa_nmda_weights == 0 else False

    event_times = list()
    if drive_type == 'poisson' and not target_syn_weights_zero:
        event_times = _create_extpois(
            t0=dynamics['tstart'],
            T=dynamics['tstop'],
            lamtha=dynamics['rate_constant'][drive_conn['target_type']],
            prng=prng)
    elif not target_syn_weights_zero and (drive_type == 'evoked' or
                                          drive_type == 'gaussian'):
        event_times = _create_gauss(
            mu=dynamics['mu'],
            sigma=dynamics['sigma'],
            numspikes=dynamics['numspikes'],
            prng=prng)
    elif drive_type == 'bursty' and not target_syn_weights_zero:
        event_times = _create_bursty_input(
            t0=dynamics['tstart'],
            t0_stdev=dynamics['tstart_std'],
            tstop=dynamics['tstop'],
            f_input=dynamics['burst_rate'],
            events_jitter_std=dynamics['burst_std'],
            repeats=dynamics['repeats'],
            events_per_cycle=dynamics['numspikes'],
            cycle_events_isi=dynamics['spike_isi'],
            prng=prng,
            prng2=prng2)

    # brute force remove non-zero times. Might result in fewer vals
    # than desired
    # values MUST be sorted for VecStim()!
    if len(event_times) > 0:
        event_times = event_times[event_times > 0]
        event_times.sort()
        event_times = event_times.tolist()

    return event_times


def drive_event_times(drive_type, target_cell_type, params, gid, trial_idx=0):
    """External spike input times.

    An external input drive to the network, i.e., one that is independent of
    the spiking output of cells in the network.

    Parameters
    ----------
    drive_type : str
        The drive type, which is one of
        'extpois' : Poisson-distributed input to proximal dendrites
        'extgauss' : Gaussian-distributed input to proximal dendrites
        'evprox' : Proximal input at specified time (or Gaussian spread)
        'evdist' : Distal input at specified time (or Gaussian spread)

        'common' : As opposed to other drive types, these have timing that is
        identical (synchronous) for all real cells in the network. Proximal
        and distal dendrites have separate parameter sets, and need not be
        synchronous. Note that not all cells classes (types) are required to
        receive 'common' input---separate conductivity values can be assigned
        to basket vs. pyramidal cells and AMPA vs. NMDA synapses
    target_cell_type : str | None
        The target cell type of the drive, e.g., 'L2_basket', 'L5_pyramidal',
        etc., or None for 'common' inputs
    params : dict
        Parameters of the external input drive, arranged into a dictionary.
    gid : int
        The cell ID.
    trial_idx : int
        The index number of the current trial of a simulation.

    Returns
    -------
    event_times : list
        The event times at which spikes occur.
    """
    prng, prng2 = _get_prng(
        seed=params['prng_seedcore'] + trial_idx,
        gid=gid,
        sync_evinput=params.get('sync_evinput', False))

    # check drive name validity, allowing substring matches ('evprox1' etc)
    valid_drives = ['extpois', 'extgauss', 'common', 'evprox', 'evdist']
    # NB check if drive_type has a valid substring, not vice versa
    matches = [f for f in valid_drives if f in drive_type]
    if len(matches) == 0:
        raise ValueError('Invalid external drive: %s' % drive_type)
    elif len(matches) > 1:
        raise ValueError('Ambiguous external drive: %s' % drive_type)

    # Return values not checked: False if all weights for given drive type
    # are zero. Designed to be silent so that zeroing input weights
    # effectively disables each.
    target_syn_weights_zero = False
    if target_cell_type in params:
        target_syn_weights_zero = (params[target_cell_type][0] <= 0.0 and
                                   params[target_cell_type][1] <= 0.0)

    all_syn_weights_zero = True
    for key in params.keys():
        if key.startswith(('L2Pyr', 'L5Pyr', 'L2Bask', 'L5Bask')):
            if params[key][0] > 0.0:
                all_syn_weights_zero = False

    event_times = list()
    if drive_type == 'extpois' and not target_syn_weights_zero:
        event_times = _create_extpois(
            t0=params['t_interval'][0],
            T=params['t_interval'][1],
            # ind 3 is frequency (lamtha))
            lamtha=params[target_cell_type][3],
            prng=prng)
    elif drive_type.startswith(('evprox', 'evdist')) and \
            target_cell_type in params:
        event_times = _create_gauss(
            mu=params['t0'],
            # ind 3 is sigma_t (stdev))
            sigma=params[target_cell_type][3],
            numspikes=int(params['numspikes']),
            prng=prng)
    elif drive_type == 'extgauss' and not target_syn_weights_zero:
        event_times = _create_gauss(
            mu=params[target_cell_type][3],
            sigma=params[target_cell_type][4],
            numspikes=50,
            prng=prng)
    elif drive_type == 'common' and not all_syn_weights_zero:
        event_times = _create_bursty_input(
            t0=params['t0'],
            t0_stdev=params['t0_stdev'],
            tstop=params['tstop'],
            f_input=params['f_input'],
            events_jitter_std=params['stdev'],
            repeats=params['repeats'],
            events_per_cycle=params['events_per_cycle'],
            cycle_events_isi=10,
            prng=prng,
            prng2=prng2)

    # brute force remove non-zero times. Might result in fewer vals
    # than desired
    # values MUST be sorted for VecStim()!
    if len(event_times) > 0:
        event_times = event_times[event_times > 0]
        event_times.sort()
        event_times = event_times.tolist()

    return event_times


def _create_extpois(*, t0, T, lamtha, prng):
    """Create poisson inputs.

    Parameters
    ----------
    t0 : float
        The start time (in ms).
    T : float
        The end time (in ms).
    lamtha : float
        The rate parameter for spike train (in Hz)
    prng : instance of RandomState
        The random state.

    Returns
    -------
    event_times : array
        The event times.
    """
    # see: http://www.cns.nyu.edu/~david/handouts/poisson.pdf
    if t0 < 0:
        raise ValueError('The start time for Poisson inputs must be'
                         f'greater than 0. Got {t0}')
    if T < t0:
        raise ValueError('The end time for Poisson inputs must be'
                         f'greater than start time. Got ({t0}, {T})')
    if lamtha <= 0.:
        raise ValueError(f'Rate must be > 0. Got {lamtha}')

    event_times = list()
    t_gen = t0
    while t_gen < T:
        t_gen += prng.exponential(1. / lamtha) * 1000.
        if t_gen < T:
            event_times.append(t_gen)

    return np.array(event_times)


def _create_gauss(*, mu, sigma, numspikes, prng):
    """Create gaussian inputs (used by extgauss and evoked).

    Parameters
    ----------
    mu : float
        The mean time of spikes.
    sigma : float
        The standard deviation.
    numspikes : float
        The number of spikes.
    prng : instance of RandomState
        The random state.

    Returns
    -------
    event_times : array
        The event times.
    """
    return prng.normal(mu, sigma, numspikes)


def _create_bursty_input(*, t0, t0_stdev, tstop, f_input,
                         events_jitter_std, repeats, events_per_cycle=2,
                         cycle_events_isi=10, prng, prng2):
    """Creates the bursty ongoing external inputs.

    Used for, e.g., for rhythmic inputs in alpha/beta generation.

    Parameters
    ----------
    t0 : float
        The start times. If -1, then randomize the start time
        of inputs uniformly between 25 ms and 125 ms.
    t0_stdev : float
        If greater than 0 and t0 != -1, randomize start time
        of inputs from a normal distribution with t0_stdev as standard
        deviation.
    tstop : float
        The stop time.
    f_input : float
        The frequency of input bursts.
    events_jitter_std : float
        The standard deviation (in ms) of each burst event.
    repeats : int
        The number of (jittered) repeats for each burst cycle.
    events_per_cycle : int
        The events per cycle. This is the spikes/burst parameter in the GUI.
        Default: 2 (doublet)
    cycle_events_isi : float
        Time between spike events within a cycle (ISI). Default: 10 ms
    prng : instance of RandomState
        The random state.
    prng2 : instance of RandomState
        The random state used for jitter to start time (see t0_stdev).

    Returns
    -------
    event_times : array
        The event times.
    """
    if t0_stdev > 0.0:
        t0 = prng2.normal(t0, t0_stdev)

    burst_period = 1000. / f_input
    burst_duration = (events_per_cycle - 1) * cycle_events_isi
    if burst_duration > burst_period:
        raise ValueError(f'Burst duration ({burst_duration}s) cannot'
                         f' be greater than burst period ({burst_period}s)'
                         'Consider increasing the spike ISI or burst rate')

    # array of mean stimulus times, starts at t0
    isi_array = np.arange(t0, tstop, burst_period)
    # array of single stimulus times -- no doublets
    t_array = prng.normal(np.repeat(isi_array, repeats), events_jitter_std)

    if events_per_cycle > 1:
        cycle = (np.arange(events_per_cycle) - (events_per_cycle - 1) / 2)
        t_array = np.ravel([t_array + cycle_events_isi * cyc for cyc in cycle])

    return t_array
