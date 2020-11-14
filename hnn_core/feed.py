"""External feed to network."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import numpy as np


# based on cdf for exp wait time distribution from unif [0, 1)
# returns in ms based on lamtha in Hz
def _t_wait(prng, lamtha):
    return -1000. * np.log(1. - prng.rand()) / lamtha


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
        Used in _create_common_input
    """
    # XXX: some param files use seed < 0 but numpy
    # does not allow this.
    if seed > 0:
        prng2 = np.random.RandomState(seed)
    else:
        prng2 = None

    if not sync_evinput:
        seed = seed + gid

    prng = np.random.RandomState(seed)
    return prng, prng2


def feed_event_times(feed_type, target_cell_type, params, gid):
    """External spike input times.

    An external input "feed" to the network, i.e., one that is independent of
    the spiking output of cells in the network.

    Parameters
    ----------
    feed_type : str
        The feed type, which is one of
        'extpois' : Poisson-distributed input to proximal dendrites
        'extgauss' : Gaussian-distributed input to proximal dendrites
        'evprox' : Proximal input at specified time (or Gaussian spread)
        'evdist' : Distal input at specified time (or Gaussian spread)

        'common' : As opposed to other feed types, these have timing that is
        identical (synchronous) for all real cells in the network. Proximal
        and distal dendrites have separate parameter sets, and need not be
        synchronous. Note that not all cells classes (types) are required to
        receive 'common' input---separate conductivity values can be assigned
        to basket vs. pyramidal cells and AMPA vs. NMDA synapses
    target_cell_type : str | None
        The target cell type of the feed, e.g., 'L2_basket', 'L5_pyramidal',
        etc., or None for 'common' inputs
    params : dict
        Parameters of the external input feed, arranged into a dictionary.
    gid : int
        The cell ID.
    """
    prng, prng2 = _get_prng(
        seed=params['prng_seedcore'],
        gid=gid,
        sync_evinput=params.get('sync_evinput', False))

    # check feed name validity, allowing substring matches ('evprox1' etc)
    valid_feeds = ['extpois', 'extgauss', 'common', 'evprox', 'evdist']
    # NB check if feed_type has a valid substring, not vice versa
    matches = [f for f in valid_feeds if f in feed_type]
    if len(matches) == 0:
        raise ValueError('Invalid external feed: %s' % feed_type)
    elif len(matches) > 1:
        raise ValueError('Ambiguous external feed: %s' % feed_type)

    # Return values not checked: False if all weights for given feed type
    # are zero. Designed to be silent so that zeroing input weights
    # effectively disables each.
    zero_ampa_nmda = False
    if target_cell_type in params:
        zero_ampa_nmda = (params[target_cell_type][0] <= 0.0 and
                          params[target_cell_type][1] <= 0.0)

    all_syn_weights_zero = True
    for key in params.keys():
        if key.startswith(('L2Pyr', 'L5Pyr', 'L2Bask', 'L5Bask')):
            if params[key][0] > 0.0:
                all_syn_weights_zero = False

    event_times = list()
    if feed_type == 'extpois' and not zero_ampa_nmda:
        event_times = _create_extpois(
            t0=params['t_interval'][0],
            T=params['t_interval'][1],
            # ind 3 is frequency (lamtha))
            lamtha=params[target_cell_type][3],
            prng=prng)
    elif feed_type.startswith(('evprox', 'evdist')) and \
            target_cell_type in params:
        event_times = _create_gauss(
            mu=params['t0'],
            # ind 3 is sigma_t (stdev))
            sigma=params[target_cell_type][3],
            numspikes=int(params['numspikes']),
            prng=prng)
    elif feed_type == 'extgauss' and not zero_ampa_nmda:
        event_times = _create_gauss(
            mu=params[target_cell_type][3],
            sigma=params[target_cell_type][4],
            numspikes=50,
            prng=prng)
    elif feed_type == 'common' and not all_syn_weights_zero:
        event_times = _create_common_input(
            distribution=params['distribution'],
            t0=params['t0'],
            t0_stdev=params['t0_stdev'],
            tstop=params['tstop'],
            f_input=params['f_input'],
            stdev=params['stdev'],
            repeats=params['repeats'],
            events_per_cycle=params['events_per_cycle'],
            prng=prng,
            prng2=prng2)

    # brute force remove non-zero times. Might result in fewer vals
    # than desired
    # values MUST be sorted for VecStim()!
    if len(event_times) > 0:
        event_times = event_times[event_times > 0]
        event_times.sort()
        event_times.tolist()

    return event_times


def _create_extpois(t0, T, lamtha, prng):
    """Create poisson inputs.

    Parameters
    ----------
    t0 : float
        The start time.
    T : float
        The end time.
    lamtha : float
        The spatial decay lambda.
    prng : instance of RandomState
        The random state.

    Returns
    -------
    event_times : array
        The event times.
    """
    if t0 < 0:
        raise ValueError('The start time for Poisson inputs must be'
                         f'greater than 0. Got {t0}')
    if T < t0:
        raise ValueError('The end time for Poisson inputs must be'
                         f'greater than start time. Got ({t0}, {T})')

    # start the initial value
    event_times = np.array([])
    if lamtha > 0.:
        t_gen = t0 + _t_wait(prng, lamtha)
        if t_gen < T:
            np.append(event_times, t_gen)

        while t_gen < T:
            # so as to not clobber confusingly base off of t_gen ...
            t_gen += _t_wait(prng, lamtha)
            if t_gen < T:
                event_times = np.append(event_times, t_gen)
    return event_times


def _create_gauss(mu, sigma, numspikes, prng):
    """Create gaussian inputs (used by extgauss and evoked).

    Parameters
    ----------
    mu : float
        The mean time of spikes.
    sigma : float
        The standard deviation. If sigma is 0,
        then return array of len numspikes
        containing only mu.
    numspikes : float
        The number of spikes.
    prng : instance of RandomState
        The random state.

    Returns
    -------
    event_times : array
        The event times.
    """
    if sigma > 0:
        event_times = prng.normal(mu, sigma, numspikes)
    else:
        # if sigma is specified at 0
        event_times = np.array([mu] * numspikes)

    return event_times


def _create_common_input(distribution, t0, t0_stdev, tstop, f_input,
                         stdev, repeats, events_per_cycle, prng, prng2):
    """Creates the common ongoing external inputs.

    Used for, e.g., for rhythmic inputs in alpha/beta generation.

    Parameters
    ----------
    distribution : str
        The distribution for each burst. One of 'normal' or 'uniform'.
    t0 : float
        The start times. If -1, then randomize the start time
        of inputs.
    t0_stdev : float
        Standard deviation of jitter to start time.
    tstop : float
        The stop time.
    f_input : float
        The frequency of input bursts.
    stdev : float
        The standard deviation.
    repeats : int
        The number of repeats.
    events_per_cycle : float
        The events per cycle. Must be 1 or 2.
    prng : instance of RandomState
        The random state.
    prng2 : instance of RandomState
        The random state used for jitter to start time (see t0_stdev).

    Returns
    -------
    event_times : array
        The event times.
    """
    if distribution not in ('normal', 'uniform'):
        raise ValueError("Indicated distribution not recognized. "
                         "Not making any common feeds.")

    # store f_input as self variable for later use if it exists in p
    if t0 == -1:
        t0 = prng.uniform(25., 125.)
    elif t0_stdev > 0.0:
        t0 = prng2.normal(t0, t0_stdev)

    if events_per_cycle != 1:
        print("events_per_cycle should be either 1 or 2, trying 2")
        events_per_cycle = 2

    if distribution == 'normal':
        # array of mean stimulus times, starts at t0
        isi_array = np.arange(t0, tstop, 1000. / f_input)
        # array of single stimulus times -- no doublets
        if stdev > 0:
            t_array = prng.normal(np.repeat(isi_array, repeats), stdev)
        else:
            t_array = isi_array
    elif distribution == 'uniform':
        n_inputs = repeats * f_input * (tstop - t0) / 1000.
        t_array = prng.uniform(t0, tstop, n_inputs)

    t_input = np.array([])
    if events_per_cycle == 2:
        # Two arrays store doublet times
        t_input_low = t_array - 5
        t_input_high = t_array + 5
        # Array with ALL stimulus times for input
        # np.append concatenates two np arrays
        t_input = np.append(t_input_low, t_input_high)
    elif events_per_cycle == 1:
        t_input = t_array

    return t_input
