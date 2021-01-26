"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>

import itertools as it
import numpy as np
from glob import glob
from copy import deepcopy
from warnings import warn

from .feed import _drive_cell_event_times
from .drives import _get_target_populations
from .drives import _check_drive_parameter_values, _check_poisson_rates
from .params import _extract_bias_specs_from_hnn_params
from .params import _extract_drive_specs_from_hnn_params
from .viz import plot_spikes_hist, plot_spikes_raster, plot_cells


def read_spikes(fname, gid_ranges=None):
    """Read spiking activity from a collection of spike trial files.

    Parameters
    ----------
    fname : str
        Wildcard expression (e.g., '<pathname>/spk_*.txt') of the
        path to the spike file(s).
    gid_ranges : dict of lists or range objects | None
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell or input IDs of different
        cell or input types. If None, each spike file must contain
        a 3rd column for spike type.

    Returns
    ----------
    cell_response : CellResponse
        An instance of the CellResponse object.
    """

    spike_times = list()
    spike_gids = list()
    spike_types = list()
    for file in sorted(glob(str(fname))):
        spike_trial = np.loadtxt(file, dtype=str)
        if spike_trial.shape[0] > 0:
            spike_times += [list(spike_trial[:, 0].astype(float))]
            spike_gids += [list(spike_trial[:, 1].astype(int))]

            # Note that legacy HNN 'spk.txt' files don't contain a 3rd column
            # for spike type. If reading a legacy version, validate that a
            # gid_dict is provided.
            if spike_trial.shape[1] == 3:
                spike_types += [list(spike_trial[:, 2].astype(str))]
            else:
                if gid_ranges is None:
                    raise ValueError("gid_ranges must be provided if spike "
                                     "types are unspecified in the "
                                     "file %s" % (file,))
                spike_types += [[]]
        else:
            spike_times += [[]]
            spike_gids += [[]]
            spike_types += [[]]

    cell_response = CellResponse(spike_times=spike_times,
                                 spike_gids=spike_gids,
                                 spike_types=spike_types)
    if gid_ranges is not None:
        cell_response.update_types(gid_ranges)

    return CellResponse(spike_times=spike_times, spike_gids=spike_gids,
                        spike_types=spike_types)


def _create_cell_coords(n_pyr_x, n_pyr_y, zdiff=1307.4):
    """Creates coordinate grid and place cells in it.

    Parameters
    ----------
    n_pyr_x : int
        The number of Pyramidal cells in x direction.
    n_pyr_y : int
        The number of Pyramidal cells in y direction.

    zdiff : float
        Expressed as a positive DEPTH of L5 relative to L2
        This is a deviation from the original, where L5 was defined at 0
        This should not change interlaminar weight/delay calculations.

    Returns
    -------
    pos_dict : dict of list of tuple (x, y, z)
        Dictionary containing coordinate positions.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        'common', or any of the elements of the list p_unique_keys

    Notes
    -----
    Common positions are all located at origin.
    Sort of a hack bc of redundancy
    """
    pos_dict = dict()

    # PYRAMIDAL CELLS
    xxrange = np.arange(n_pyr_x)
    yyrange = np.arange(n_pyr_y)

    pos_dict['L2_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [0])]
    pos_dict['L5_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [zdiff])]

    # BASKET CELLS
    xzero = np.arange(0, n_pyr_x, 3)
    xone = np.arange(1, n_pyr_x, 3)
    # split even and odd y vals
    yeven = np.arange(0, n_pyr_y, 2)
    yodd = np.arange(1, n_pyr_y, 2)
    # create general list of x,y coords and sort it
    coords = [pos for pos in it.product(
        xzero, yeven)] + [pos for pos in it.product(xone, yodd)]
    coords_sorted = sorted(coords, key=lambda pos: pos[1])
    # append the z value for position for L2 and L5
    # print(len(coords_sorted))

    pos_dict['L2_basket'] = [pos_xy + (0,) for
                             pos_xy in coords_sorted]
    pos_dict['L5_basket'] = [
        pos_xy + (zdiff,) for pos_xy in coords_sorted]

    # ORIGIN
    # origin's z component isn't really used in
    # calculating distance functions from origin
    # these will be forced as ints!
    origin_x = xxrange[int((len(xxrange) - 1) // 2)]
    origin_y = yyrange[int((len(yyrange) - 1) // 2)]
    origin_z = np.floor(zdiff / 2)
    origin = (origin_x, origin_y, origin_z)

    # save the origin for adding external drives later
    pos_dict['origin'] = origin

    return pos_dict


class Network(object):
    """The Network class.

    Parameters
    ----------
    params : dict
        The parameters to use for constructing the network.
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool
        Set to True by default to enable matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.

    Attributes
    ----------
    params : dict
        The parameters of the network
    cellname_list : list
        The names of real cell types in the network (e.g. 'L2_basket')
    gid_ranges : dict
        A dictionary of unique identifiers of each real and artificial cell
        in the network. Every cell type is represented by a key read from
        cellname_list, followed by keys read from external_drives. The value
        of each key is a range of ints, one for each cell in given category.
        Examples: 'L2_basket': range(0, 270), 'evdist1': range(272, 542), etc
    pos_dict : dict
        Dictionary containing the coordinate positions of all cells.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        or any external drive name
    cell_response : CellResponse
        An instance of the CellResponse object.
    external_drives : dict (keys: drive names) of dict (keys: parameters)
        The external driving inputs to the network. Drives are added by
        defining their spike-time dynamics, and their connectivity to the real
        cells of the network. Event times are instantiated before simulation,
        and are stored under the ``'events'``-key (list of list; first
        index for trials, second for event time lists for each drive cell).
    external_biases : dict of dict (bias parameters for each cell type)
        The parameters of bias inputs to cell somata, e.g., tonic current clamp
    """

    def __init__(self, params, add_drives_from_params=False,
                 legacy_mode=True):
        # Save the parameters used to create the Network
        self.params = params
        # Initialise a dictionary of cell ID's, which get used when the
        # network is constructed ('built') in NetworkBuilder
        # We want it to remain in each Network object, so that the user can
        # interrogate a built and simulated net. In addition, CellResponse is
        # attached to a Network during simulation---Network is the natural
        # place to keep this information
        self.gid_ranges = dict()
        self._n_gids = 0  # utility: keep track of last GID

        # XXX this can be removed once tests are made independent of HNN GUI
        # creates nc_dict-entries for ALL cell types
        self._legacy_mode = legacy_mode

        # Create array of equally sampled time points for simulating currents
        # NB (only) used to initialise self.cell_response._times
        times = np.arange(0., params['tstop'] + params['dt'], params['dt'])
        # Create CellResponse object, initialised with simulation time points
        self.cell_response = CellResponse(times=times)

        # Source list of names, first real ones only!
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]

        # external drives and biases
        self.external_drives = dict()
        self.external_biases = dict()

        # contents of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self.pos_dict = _create_cell_coords(n_pyr_x=self.params['N_pyr_x'],
                                            n_pyr_y=self.params['N_pyr_y'],
                                            zdiff=1307.4)
        # Every time pos_dict is updated, gid_ranges must be updated too
        self._update_gid_ranges()

        # set n_cells, EXCLUDING Artificial ones
        self.n_cells = sum(len(self.pos_dict[src]) for src in
                           self.cellname_list)

        if add_drives_from_params:
            drive_specs = _extract_drive_specs_from_hnn_params(
                self.params, self.cellname_list)
            bias_specs = _extract_bias_specs_from_hnn_params(
                self.params, self.cellname_list)

            for drive_name in sorted(drive_specs.keys()):  # order matters
                specs = drive_specs[drive_name]
                if specs['type'] == 'evoked':
                    self.add_evoked_drive(
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
                    self.add_poisson_drive(
                        drive_name, tstart=specs['dynamics']['tstart'],
                        tstop=specs['dynamics']['tstop'],
                        rate_constant=specs['dynamics']['rate_constant'],
                        weights_ampa=specs['weights_ampa'],
                        weights_nmda=specs['weights_nmda'],
                        location=specs['location'], seedcore=specs['seedcore'],
                        synaptic_delays=specs['synaptic_delays'],
                        space_constant=specs['space_constant'])
                elif specs['type'] == 'gaussian':
                    self.add_evoked_drive(  # 'gaussian' is just evoked
                        drive_name, mu=specs['dynamics']['mu'],
                        sigma=specs['dynamics']['sigma'],
                        numspikes=specs['dynamics']['numspikes'],
                        weights_ampa=specs['weights_ampa'],
                        weights_nmda=specs['weights_nmda'],
                        location=specs['location'], seedcore=specs['seedcore'],
                        synaptic_delays=specs['synaptic_delays'],
                        space_constant=specs['space_constant'])
                elif specs['type'] == 'bursty':
                    self.add_bursty_drive(
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
                self.add_tonic_bias(
                    cell_type=cellname,
                    amplitude=bias_specs['tonic'][cellname]['amplitude'],
                    t0=bias_specs['tonic'][cellname]['t0'],
                    T=bias_specs['tonic'][cellname]['T'])

            self._instantiate_drives(n_trials=self.params['N_trials'])

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.params['N_pyr_x'], self.params['N_pyr_y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (len(self.pos_dict['L2_basket']),
                 len(self.pos_dict['L5_basket'])))
        return '<%s | %s>' % (class_name, s)

    def copy(self):
        """Return a copy of the Network instance

        The returned copy retains the intrinsic connectivity between cells, as
        well as those of any external drives or biases added to the network.
        The parameters of drive dynamics are also retained, but the
        instantiated ``events`` of the drives are cleared. This allows
        iterating over the values defining drive dynamics, without the need to
        re-define connectivity.

        Returns
        -------
        net_copy : instance of Network
            A copy of the instance with previous simulation results and
            ``events`` of external drives removed.
        """
        net_copy = deepcopy(self)
        net_copy.cell_response = CellResponse(times=self.cell_response._times)
        net_copy._reset_drives()
        return net_copy

    def add_evoked_drive(self, name, *, mu, sigma, numspikes,
                         sync_within_trial=False, location,
                         weights_ampa=None, weights_nmda=None,
                         space_constant=3., synaptic_delays=0.1, seedcore=2):
        """Add an 'evoked' external drive to the network

        Parameters
        ----------
        name : str
            Unique name for the drive
        mu : float
            Mean of Gaussian event time distribution
        sigma : float
            Standard deviation of event time distribution
        numspikes : int
            Number of spikes at each target cell
        sync_within_trial : bool
            If True, the target cells receive each numspike synchronously. By
            default (False), spike times arriving at each target cell are
            sampled independently using the Gaussian parameteres (mu, sigma).
        location : str
            Target location of synapses ('distal' or 'proximal')
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : float or dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        seedcore : int
            Optional initial seed for random number generator (default: 2).
        """
        if not self._legacy_mode:
            _check_drive_parameter_values('evoked', sigma=sigma,
                                          numspikes=numspikes)

        drive = _NetworkDrive()
        drive['type'] = 'evoked'
        if name == 'extgauss':
            drive['type'] = 'gaussian'  # XXX needed to pass legacy tests!
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(mu=mu, sigma=sigma, numspikes=numspikes,
                                 sync_within_trial=sync_within_trial)
        drive['events'] = list()

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, synaptic_delays)

    def add_poisson_drive(self, name, *, tstart=0, tstop=None, rate_constant,
                          location, weights_ampa=None, weights_nmda=None,
                          space_constant=100., synaptic_delays=0.1,
                          seedcore=2):
        """Add a Poisson-distributed external drive to the network

        Parameters
        ----------
        name : str
            Unique name for the drive
        tstart : float
            Start time of Poisson-distributed spike train (default: 0)
        tstop : float
            End time of the spike train (defaults to None: tstop is set to the
            end of the simulation)
        rate_constant : float or dict of floats
            Rate constant (lambda) of the renewal-process generating the
            samples. If a float is provided, the same rate constant is applied
            to each target cell type. Cell type-specific values may be
            provided as a dictionary, in which a key must be present for each
            cell type with non-zero AMPA or NMDA weights.
        location : str
            Target location of synapses ('distal' or 'proximal')
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : float or dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weigths and delays within the simulated column
        seedcore : int
            Optional initial seed for random number generator (default: 2).
        """
        sim_end_time = self.cell_response.times[-1]
        if tstop is None:
            tstop = sim_end_time

        if not self._legacy_mode:
            _check_drive_parameter_values('Poisson', tstart=tstart,
                                          tstop=tstop,
                                          sim_end_time=sim_end_time)
            target_populations = _get_target_populations(weights_ampa,
                                                         weights_nmda)[0]
            _check_poisson_rates(rate_constant, target_populations,
                                 self.cellname_list)

        drive = _NetworkDrive()
        drive['type'] = 'poisson'
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(tstart=tstart, tstop=tstop,
                                 rate_constant=rate_constant)
        drive['events'] = list()
        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, synaptic_delays)

    def add_bursty_drive(self, name, *, tstart=0, tstart_std=0, tstop=None,
                         location, burst_rate, burst_std=0, numspikes=2,
                         spike_isi=10, repeats=1, weights_ampa=None,
                         weights_nmda=None, synaptic_delays=0.1,
                         space_constant=100., seedcore=2,
                         distribution='normal'):
        """Add a bursty (rhythmic) external drive to all cells of the network

        Parameters
        ----------
        name : str
            Unique name for the drive
        tstart : float
            Start time of the burst trains (default: 0)
        tstart_std : float
            If greater than 0, randomize start time with standard deviation
            tstart_std (unit: ms). Effectively jitters start time across
            multiple trials.
        tstop : float
            End time of burst trains (defaults to None: tstop is set to the
            end of the simulation)
        burst_rate : float
            The mean rate at which cyclic bursts occur (unit: Hz)
        burst_std : float
            The standard deviation of the burst occurrence on each cycle
            (unit: ms). Default: 0 ms
        numspikes : int
            The number of spikes in a burst. This is the spikes/burst parameter
            in the GUI. Default: 2 (doublet)
        spike_isi : float
            Time between spike events within a cycle (ISI). Default: 10 ms
        repeats : int
            The number of bursts per cycle. Default: 1
        location : str
            Target location of synapses ('distal' or 'proximal')
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : float or dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        seedcore : int
            Optional initial seed for random number generator (default: 2).
        distribution : str
            Must be 'normal' (will be deprecated in a future release).
        """
        sim_end_time = self.cell_response.times[-1]
        if tstop is None:
            tstop = sim_end_time
        if not self._legacy_mode:
            _check_drive_parameter_values('bursty', tstart=tstart, tstop=tstop,
                                          sim_end_time=sim_end_time,
                                          sigma=tstart_std, location=location)
            _check_drive_parameter_values('bursty', sigma=burst_std,
                                          numspikes=numspikes,
                                          spike_isi=spike_isi,
                                          burst_rate=burst_rate)

        drive = _NetworkDrive()
        drive['type'] = 'bursty'
        drive['cell_specific'] = False
        drive['seedcore'] = seedcore

        # XXX distribution='uniform' should be deprecated as it is used nowhere
        # alternatively, create a new drive-type for it
        drive['dynamics'] = dict(distribution=distribution, tstart=tstart,
                                 tstart_std=tstart_std, tstop=tstop,
                                 burst_rate=burst_rate, burst_std=burst_std,
                                 numspikes=numspikes, spike_isi=spike_isi,
                                 repeats=repeats)
        drive['events'] = list()

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, synaptic_delays,
                           cell_specific=False)

    def _attach_drive(self, name, drive, weights_ampa, weights_nmda, location,
                      space_constant, synaptic_delays, cell_specific=True):
        """Attach a drive to network based on connectivity information

        Parameters
        ----------
        name : str
            Name of drive (must be unique)
        drive : instance of _NetworkDrive
            Collection of parameters defining the dynamics of the drive
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        location : str
            Target location of synapses ('distal' or 'proximal')
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        synaptic_delays : dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant
        cell_specific : bool
            Whether each cell has unique connection parameters (default: True)
            or all cells have common connections to a global (single) drive.

        Attached drive is stored in self.external_drives[name]
        self.pos_dict is updated, and self._update_gid_ranges() called
        """
        if name in self.external_drives:
            raise ValueError(f"Drive {name} already defined")
        if location not in ['distal', 'proximal']:
            raise ValueError("Allowed drive target locations are: 'distal', "
                             f"and 'proximal', got {location}")
        # allow passing weights as None, convert to dict here
        target_populations, weights_ampa, weights_nmda = \
            _get_target_populations(weights_ampa, weights_nmda)

        # weights passed must correspond to cells in the network
        if not target_populations.issubset(set(self.cellname_list)):
            raise ValueError('Allowed drive target cell types are: ',
                             f'{self.cellname_list}')

        weights_by_receptor = {'ampa': weights_ampa, 'nmda': weights_nmda}
        if isinstance(synaptic_delays, dict):
            for receptor in ['ampa', 'nmda']:
                # synaptic_delays must be defined for all cell types for which
                # either AMPA or NMDA weights are non-zero
                if not (set(weights_by_receptor[receptor].keys()).issubset(
                        set(synaptic_delays.keys()))):
                    raise ValueError(
                        'synaptic_delays is either a common float or needs '
                        'to be specified as a dict for each cell type')

        # this is needed to keep the drive GIDs identical to those in HNN,
        # e.g., 'evdist1': range(272, 542), even when no L5_basket cells
        # are targeted (event times lists are empty). The logic in
        # _connect_celltypes is hard-coded to use these implict gid ranges
        if self._legacy_mode:
            # XXX tests must match HNN GUI output
            target_populations = self.cellname_list
        elif len(target_populations) == 0:
            warn('No AMPA or NMDA weights > 0')

        drive['name'] = name  # for easier for-looping later
        drive['target_types'] = target_populations  # for _connect_celltypes

        drive['conn'], src_gid_ran = self._create_drive_conns(
            target_populations, weights_by_receptor, location,
            space_constant, synaptic_delays, cell_specific=cell_specific)

        # Must remember to update the GID ranges based on pos_dict!
        self.pos_dict[name] = [self.pos_dict['origin'] for _ in src_gid_ran]

        # NB _update_gid_ranges checks external_drives[name] for drives!
        self.external_drives[name] = drive
        # Every time pos_dict is updated, gid_ranges must be updated too
        self._update_gid_ranges()

    def _create_drive_conns(self, target_populations, weights_by_receptor,
                            location, space_constant, synaptic_delays,
                            cell_specific=True):
        """Create parameter dictionary defining how drive connects to network

        Parameters
        ----------
        target_populations : list
            Cell names/types to attach the drive to
        weights_by_receptor : dict (keys: 'ampa' and 'nmda')
            Synaptic weights (in uS) for each receptor type
        location : str
            Target location of synapses ('distal' or 'proximal')
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant
        cell_specific : bool
            Whether each cell has unique connection parameters (default: True)
            or all cells have common connections to a global (single) drive.

        Returns
        -------
        drive_conn_by_cell : dict
            Keys are target_populations, each item is itself a dict with keys
            relevant to how the drive connects to the rest of the network.
        src_gid_ran : range
            For convenience, return back the range of GIDs associated with
            all the driving units (they become _ArtificialCells later)

        Note
        ----
        drive_conn : dict
            'target_gids': range of target cell GIDs
            'target_type': target cell type (e.g. 'L2_basket')
            'src_gids': range of source (artificial) cell GIDs
            'location': 'distal' or 'proximal'
            'ampa' and 'nmda': dict
                'A_weight': synaptic weight
                'A_delay':  synaptic delay (used with space constant)
                'lamtha': space constant
        """
        drive_conn_by_cell = dict()
        src_gid_ran_begin = self._n_gids

        if cell_specific:
            for cellname in target_populations:
                drive_conn = dict()  # NB created inside target_pop-loop
                drive_conn['location'] = location

                drive_conn['target_gids'] = self.gid_ranges[cellname]
                # NB list! This is used later in _parnet_connect
                drive_conn['target_type'] = cellname
                drive_conn['src_gids'] = range(
                    self._n_gids,
                    self._n_gids + len(drive_conn['target_gids']))
                self._n_gids += len(drive_conn['target_gids'])
                drive_conn_by_cell[cellname] = drive_conn
        else:
            drive_conn = dict()
            drive_conn['location'] = location

            # NB list! This is used later in _parnet_connect
            drive_conn['src_gids'] = [self._n_gids]
            self._n_gids += 1

            drive_conn['target_gids'] = list()  # fill in below
            for cellname in target_populations:
                drive_conn['target_gids'].extend(self.gid_ranges[cellname])
                drive_conn['target_type'] = cellname
                drive_conn_by_cell[cellname] = drive_conn

        for cellname in target_populations:
            for receptor, weights in weights_by_receptor.items():
                drive_conn_by_cell[cellname][receptor] = dict()
                if cellname in weights:
                    drive_conn_by_cell[cellname][receptor][
                        'lamtha'] = space_constant
                    if isinstance(synaptic_delays, float):
                        drive_conn_by_cell[cellname][receptor][
                            'A_delay'] = synaptic_delays
                    elif isinstance(synaptic_delays, dict):
                        drive_conn_by_cell[cellname][receptor][
                            'A_delay'] = synaptic_delays[cellname]
                    drive_conn_by_cell[cellname][receptor][
                        'A_weight'] = weights[cellname]

        return drive_conn_by_cell, range(src_gid_ran_begin, self._n_gids)

    def _reset_drives(self):
        # reset every time called again, e.g., from dipole.py or in self.copy()
        for drive_name in self.external_drives.keys():
            self.external_drives[drive_name]['events'] = list()

    def _instantiate_drives(self, n_trials=1):
        """Creates drive_event_times vectors for all drives and all trials

        Parameters
        ----------
        n_trials : int
            Number of trials to create events for (default: 1)

        NB this must be a separate method because dipole.py:simulate_dipole
        accepts an n_trials-argument, which overrides the N_trials-parameter
        used at intialisation time. The good news is that only the event_times
        need to be recalculated, all the GIDs etc remain the same.
        """
        self._reset_drives()

        # each trial needs unique event time vectors
        for trial_idx in range(n_trials):
            for drive in self.external_drives.values():
                event_times = list()  # new list for each trial and drive

                # loop over drive 'cells' and create event times for each
                for this_cell_drive_conn in drive['conn'].values():
                    for drive_cell_gid in this_cell_drive_conn['src_gids']:
                        event_times.append(_drive_cell_event_times(
                            drive['type'], this_cell_drive_conn,
                            drive['dynamics'], trial_idx=trial_idx,
                            drive_cell_gid=drive_cell_gid,
                            seedcore=drive['seedcore'])
                        )
                    # only create one event_times list for globals
                    if not drive['cell_specific']:
                        break  # loop over drive['conn'].values

                # 'events': list (trials) of list (cells) of list (events)
                self.external_drives[
                    drive['name']]['events'].append(event_times)

    def add_tonic_bias(self, *, cell_type=None, amplitude=None,
                       t0=None, T=None):
        """Attach parameters of tonic biasing input for a given cell type.

        Parameters
        ----------
        cell_type : str
            The cell type whose cells will get the tonic input.
            Valid inputs are those in `net.cellname_list`.
        amplitude : float
            The amplitude of the input.
        t0 : float
            The start time of tonic input (in ms). Default: 0 (beginning of
            simulation).
        T : float
            The end time of tonic input (in ms). Default: end of simulation.
        """
        if (cell_type is None or amplitude is None):
            raise ValueError('cell_type and amplitude must be defined'
                             f', got {cell_type}, {amplitude}')
        if 'tonic' not in self.external_biases:
            self.external_biases['tonic'] = dict()
        if cell_type in self.external_biases['tonic']:
            raise ValueError(f'Tonic bias already defined for {cell_type}')

        if t0 is None:
            t0 = 0
        tstop = self.cell_response.times[-1]
        if T is None:
            T = tstop
        if T < 0.:
            raise ValueError('End time of tonic input cannot be negative')
        if T > tstop:
            raise ValueError(f'End time of tonic input cannot exceed '
                             f'simulation end time {tstop}. Got {T}.')
        if cell_type not in self.cellname_list:
            raise ValueError(f'cell_type must be one of {self.cellname_list}'
                             f'. Got {cell_type}')
        duration = T - t0
        if duration < 0.:
            raise ValueError('Duration of tonic input cannot be negative')

        self.external_biases['tonic'][cell_type] = {
            'amplitude': amplitude,
            't0': t0,
            'T': T
        }

    def _update_gid_ranges(self):
        """Creates gid ranges from scratch every time called.

        Any method that adds real or artificial cells to the network must
        call this to update the list of GIDs. Note that it's based on the
        content of pos_dict and the lists of real and artificial cell names.
        """
        # if external drives dict is empty, list will also be empty
        ext_drive_names = list(self.external_drives.keys())
        gid_lims = [0]  # start and end gids per cell type
        src_types = self.cellname_list + ext_drive_names
        for idx, src_type in enumerate(src_types):
            n_srcs = len(self.pos_dict[src_type])
            gid_lims.append(gid_lims[idx] + n_srcs)
            self.gid_ranges[src_type] = range(gid_lims[idx],
                                              gid_lims[idx + 1])
        self._n_gids = gid_lims[idx + 1]

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        for gidtype, gids in self.gid_ranges.items():
            if gid in gids:
                return gidtype

    def _get_src_type_and_pos(self, gid):
        """Source type, position and whether it's a cell or artificial feed"""

        # get type of cell and pos via gid
        src_type = self.gid_to_type(gid)
        type_pos_ind = gid - self.gid_ranges[src_type][0]
        src_pos = self.pos_dict[src_type][type_pos_ind]

        return src_type, src_pos, src_type in self.cellname_list

    def plot_cells(self, ax=None, show=True):
        """Plot the cells using Network.pos_dict.

        Parameters
        ----------
        ax : instance of matplotlib Axes3D | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_cells(net=self, ax=ax, show=show)


class _NetworkDrive(dict):
    """A class for containing the parameters of external drives

    Class instances are essentially dictionaries, with keys described below
    as 'attributes'. For example, drive['events'] contains the spike times of
    exogeneous inputs.

    Attributes
    ----------
    name : str
        Name of drive (must be unique)
    type : str
        Examples: 'evoked', 'gaussian', 'poisson', 'bursty'
    events : list of lists
        List of spike time lists. First index is of length n_trials. Second
        index is over the 'artificial' cells associated with this drive.
    cell_specific : bool
        Whether each cell has unique connection parameters (default: True)
        or all cells have common connections to a global (single) drive.
    seedcore : int
        Optional initial seed for random number generator
        Each artificial drive cell has seed = seedcore + gid
    target_types : set or list of str
        Names of cell types targeted by this drive (must be subset of
        net.cellname_list).
    dynamics : dict
        Parameters describing how the temporal dynamics of spike trains in the
        drive. The keys are specific to the type of drive ('evoked', 'bursty',
        etc.). See the drive add-methods in Network for details.
    conn : dict
        Parameters describing how the drive connects to the network. Keys are:
        'target_gids': range of target cell GIDs;
        'target_type': target cell type (e.g. 'L2_basket');
        'src_gids': range of source (artificial) cell GIDs;
        'location': 'distal' or 'proximal';
    conn['ampa'] and conn['nmda']: dict
        Sub-dicts specifying the synaptic weights:
        'A_weight': synaptic weight;
        'A_delay': synaptic delay at d=0 (used with space constant);
        'lamtha': space constant
    """
    def __repr__(self):
        entr = f"<External drive '{self['name']}'"
        if 'type' in self.keys():
            entr += f"\ndrive class: {self['type']}"
            entr += f"\ntarget cell types: {self['target_types']}"
            entr += f"\ncell-specific: {self['cell_specific']}"
            entr += "\ndynamic parameters:"
            for key, val in self['dynamics'].items():
                entr += f"\n\t{key}: {val}"
        if len(self['events']) > 0:
            plurl = 's' if len(self['events']) > 1 else ''
            entr += ("\nevent times instantiated for "
                     f"{len(self['events'])} trial{plurl}")
        entr += '>'
        return entr


class CellResponse(object):
    """The CellResponse class.

    Parameters
    ----------
    spike_times : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network().gid_ranges.

    Attributes
    ----------
    spike_times : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network::gid_ranges.
    vsoma : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing somatic voltages.
    isoma : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing somatic currents.

    times : numpy array
        Array of time points for samples in continuous data.
        This includes vsoma and isoma.

    Methods
    -------
    update_types(gid_ranges)
        Update spike types in the current instance of CellResponse.
    plot(ax=None, show=True)
        Plot and return a matplotlib Figure object showing the
        aggregate network spiking activity according to cell type.
    mean_rates(tstart, tstop, gid_ranges, mean_type='all')
        Calculate mean firing rate for each cell type. Specify
        averaging method with mean_type argument.
    write(fname)
        Write spiking activity to a collection of spike trial files.
    """

    def __init__(self, spike_times=None, spike_gids=None, spike_types=None,
                 times=None):
        if spike_times is None:
            spike_times = list()
        if spike_gids is None:
            spike_gids = list()
        if spike_types is None:
            spike_types = list()

        # Validate arguments
        arg_names = ['spike_times', 'spike_gids', 'spike_types']
        for arg_idx, arg in enumerate([spike_times, spike_gids, spike_types]):
            # Validate outer list
            if not isinstance(arg, list):
                raise TypeError('%s should be a list of lists'
                                % (arg_names[arg_idx],))
            # If arg is not an empty list, validate inner list
            for trial_list in arg:
                if not isinstance(trial_list, list):
                    raise TypeError('%s should be a list of lists'
                                    % (arg_names[arg_idx],))
            # Set the length of 'spike_times' as a references and validate
            # uniform length
            if arg == spike_times:
                n_trials = len(spike_times)
            if len(arg) != n_trials:
                raise ValueError('spike times, gids, and types should be '
                                 'lists of the same length')
        self._spike_times = spike_times
        self._spike_gids = spike_gids
        self._spike_types = spike_types
        self._vsoma = list()
        self._isoma = list()
        if times is not None:
            if not isinstance(times, np.ndarray):
                raise TypeError("'times' is an np.ndarray of simulation times")
        self._times = times

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._spike_times)
        return '<%s | %d simulation trials>' % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, CellResponse):
            return NotImplemented
        # Round each time element
        times_self = [[round(time, 3) for time in trial]
                      for trial in self._spike_times]
        times_other = [[round(time, 3) for time in trial]
                       for trial in other._spike_times]
        return (times_self == times_other and
                self._spike_gids == other._spike_gids and
                self._spike_types == other._spike_types)

    def __getitem__(self, gid_item):
        """Returns a CellResponse object with a copied subset filtered by gid.

        Parameters
        ----------
        gid_item : int | slice
            Subset of gids .

        Returns
        -------
        cell_response : instance of CellResponse
            See below for use cases.
        """

        if isinstance(gid_item, slice):
            gid_item = np.arange(gid_item.stop)[gid_item]
        elif isinstance(gid_item, list):
            gid_item = np.array(gid_item)
        elif isinstance(gid_item, np.ndarray):
            if gid_item.ndim > 1:
                raise ValueError("ndarray cannot exceed 1 dimension")
            else:
                pass
        elif isinstance(gid_item, int):
            gid_item = np.array([gid_item])
        else:
            raise TypeError("indices must be int, slice, or array-like, "
                            f"not {type(gid_item).__name__}")

        if not np.issubdtype(gid_item.dtype, np.integer):
            raise TypeError("gids must be of dtype int, "
                            f"not {gid_item.dtype.name}")

        n_trials = len(self._spike_times)
        times_slice = list()
        gids_slice = list()
        types_slice = list()
        vsoma_slice = list()
        isoma_slice = list()
        for trial_idx in range(n_trials):
            gid_mask = np.in1d(self._spike_gids[trial_idx], gid_item)
            times_trial = np.array(
                self._spike_times[trial_idx])[gid_mask].tolist()
            gids_trial = np.array(
                self._spike_gids[trial_idx])[gid_mask].tolist()
            types_trial = np.array(
                self._spike_types[trial_idx])[gid_mask].tolist()

            vsoma_trial = {gid: self._vsoma[trial_idx][gid] for gid in gid_item
                           if gid in self._vsoma[trial_idx].keys()}

            isoma_trial = {gid: self._isoma[trial_idx][gid] for gid in gid_item
                           if gid in self._isoma[trial_idx].keys()}

            times_slice.append(times_trial)
            gids_slice.append(gids_trial)
            types_slice.append(types_trial)
            vsoma_slice.append(vsoma_trial)
            isoma_slice.append(isoma_trial)

        cell_response_slice = CellResponse(spike_times=times_slice,
                                           spike_gids=gids_slice,
                                           spike_types=types_slice)
        cell_response_slice._vsoma = vsoma_slice
        cell_response_slice._isoma = isoma_slice

        return cell_response_slice

    @property
    def spike_times(self):
        return self._spike_times

    @property
    def spike_gids(self):
        return self._spike_gids

    @property
    def spike_types(self):
        return self._spike_types

    @property
    def vsoma(self):
        return self._vsoma

    @property
    def isoma(self):
        return self._isoma

    @property
    def times(self):
        return self._times

    def update_types(self, gid_ranges):
        """Update spike types in the current instance of CellResponse.

        Parameters
        ----------
        gid_ranges : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        """

        # Validate gid_ranges
        all_gid_ranges = list(gid_ranges.values())
        for item_idx_1 in range(len(all_gid_ranges)):
            for item_idx_2 in range(item_idx_1 + 1, len(all_gid_ranges)):
                gid_set_1 = set(all_gid_ranges[item_idx_1])
                gid_set_2 = set(all_gid_ranges[item_idx_2])
                if not gid_set_1.isdisjoint(gid_set_2):
                    raise ValueError('gid_ranges should contain only disjoint '
                                     'sets of gid values')

        spike_types = list()
        for trial_idx in range(len(self._spike_times)):
            spike_types_trial = np.empty_like(self._spike_times[trial_idx],
                                              dtype='<U36')
            for gidtype, gids in gid_ranges.items():
                spike_gids_mask = np.in1d(self._spike_gids[trial_idx], gids)
                spike_types_trial[spike_gids_mask] = gidtype
            spike_types += [list(spike_types_trial)]
        self._spike_types = spike_types

    def mean_rates(self, tstart, tstop, gid_ranges, mean_type='all'):
        """Mean spike rates (Hz) by cell type.

        Parameters
        ----------
        tstart : int | float | None
            Value defining the start time of all trials.
        tstop : int | float | None
            Value defining the stop time of all trials.
        gid_ranges : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        mean_type : str
            'all' : Average over trials and cells
                Returns mean rate for cell types
            'trial' : Average over cell types
                Returns trial mean rate for cell types
            'cell' : Average over individual cells
                Returns trial mean rate for individual cells

        Returns
        -------
        spike_rate : dict
            Dictionary with keys 'L5_pyramidal', 'L5_basket', etc.
        """
        cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
        spike_rates = dict()

        if mean_type not in ['all', 'trial', 'cell']:
            raise ValueError("Invalid mean_type. Valid arguments include "
                             f"'all', 'trial', or 'cell'. Got {mean_type}")

        # Validate tstart, tstop
        if not isinstance(tstart, (int, float)) or not isinstance(
                tstop, (int, float)):
            raise ValueError('tstart and tstop must be of type int or float')
        elif tstop <= tstart:
            raise ValueError('tstop must be greater than tstart')

        for cell_type in cell_types:
            cell_type_gids = np.array(gid_ranges[cell_type])
            n_trials, n_cells = len(self._spike_times), len(cell_type_gids)
            gid_spike_rate = np.zeros((n_trials, n_cells))

            trial_data = zip(self._spike_types, self._spike_gids)
            for trial_idx, (spike_types, spike_gids) in enumerate(trial_data):
                trial_type_mask = np.in1d(spike_types, cell_type)
                gids, gid_counts = np.unique(np.array(
                    spike_gids)[trial_type_mask], return_counts=True)

                gid_spike_rate[trial_idx, np.in1d(cell_type_gids, gids)] = (
                    gid_counts / (tstop - tstart)) * 1000

            if mean_type == 'all':
                spike_rates[cell_type] = np.mean(
                    gid_spike_rate.mean(axis=1))
            if mean_type == 'trial':
                spike_rates[cell_type] = np.mean(
                    gid_spike_rate, axis=1).tolist()
            if mean_type == 'cell':
                spike_rates[cell_type] = [gid_trial_rate.tolist()
                                          for gid_trial_rate in gid_spike_rate]

        return spike_rates

    def plot_spikes_raster(self, ax=None, show=True):
        """Plot the aggregate spiking activity according to cell type.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure object.
        """
        return plot_spikes_raster(cell_response=self, ax=ax, show=show)

    def plot_spikes_hist(self, ax=None, spike_types=None, show=True):
        """Plot the histogram of spiking activity across trials.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        spike_types: string | list | dictionary | None
            String input of a valid spike type is plotted individually.
                Ex: 'common', 'evdist', 'evprox', 'extgauss', 'extpois'
            List of valid string inputs will plot each spike type individually.
                Ex: ['common', 'evdist']
            Dictionary of valid lists will plot list elements as a group.
                Ex: {'Evoked': ['evdist', 'evprox'], 'External': ['extpois']}
            If None, all input spike types are plotted individually if any
            are present. Otherwise spikes from all cells are plotted.
            Valid strings also include leading characters of spike types
                Example: 'ext' is equivalent to ['extgauss', 'extpois']
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_spikes_hist(
            self, ax=ax, spike_types=spike_types, show=show)

    def write(self, fname):
        """Write spiking activity per trial to a collection of files.

        Parameters
        ----------
        fname : str
            String format (e.g., '<pathname>/spk_%d.txt') of the
            path to the output spike file(s).

        Outputs
        -------
        A tab separated txt file for each trial where rows
            correspond to spikes, and columns correspond to
            1) spike time (s),
            2) spike gid, and
            3) gid type
        """

        for trial_idx in range(len(self._spike_times)):
            with open(str(fname) % (trial_idx,), 'w') as f:
                for spike_idx in range(len(self._spike_times[trial_idx])):
                    f.write('{:.3f}\t{}\t{}\n'.format(
                        self._spike_times[trial_idx][spike_idx],
                        int(self._spike_gids[trial_idx][spike_idx]),
                        self._spike_types[trial_idx][spike_idx]))
