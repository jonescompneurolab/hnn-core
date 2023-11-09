# import itertools as it
# from copy import deepcopy
# from collections import OrderedDict

# import numpy as np
# import warnings

# from .drives import _drive_cell_event_times
# from .drives import _get_target_properties, _add_drives_from_params
# from .drives import _check_drive_parameter_values, _check_poisson_rates
# from .cells_default import pyramidal, basket
# from .params import _long_name, _short_name
# from .viz import plot_cells
# from .externals.mne import _validate_type, _check_option
# from .extracellular import ExtracellularArray
# from .check import _check_gids, _gid_to_type, _string_input_to_list


from netpyne import specs, sim


class NetPyne_Model(object):
    """The NetPyne_Model class.

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
    cell_types : dict
        Dictionary containing names of real cell types in the network
        (e.g. 'L2_basket') as keys and corresponding Cell instances as values.
        The Cell instance associated with a given key is used as a template
        for the other cells of its type in the population.
    gid_ranges : dict
        A dictionary of unique identifiers of each real and artificial cell
        in the network. Every cell type is represented by a key read from
        cell_types, followed by keys read from external_drives. The value
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
    connectivity : list of dict
        List of dictionaries specifying each cell-cell and drive-cell
        connection
    rec_arrays : dict
        Stores electrode position information and voltages recorded by them
        for extracellular potential measurements. Multiple electrode arrays
        may be defined as unique keys. The values of the dictionary are
        instances of :class:`hnn_core.extracellular.ExtracellularArray`.
    threshold : float
        Firing threshold of all cells.
    delay : float
        Synaptic delay in ms.

    Notes
    -----
    ``net = jones_2009_model(params)`` is the reccomended path for creating a
    network. Instantiating the network as ``net = Network(params)`` will
    produce a network with no cell-to-cell connections. As such,
    connectivity information contained in ``params`` will be ignored.
    """

    def __init__(self, params, add_drives_from_params=False,
                 legacy_mode=False):
        # Network parameters
        netParams = specs.NetParams()  # object of class NetParams to store the network parameters

        self.netParams = netParams

         ## Cell parameters/rules
        PYRcell = {'secs': {}}
        PYRcell['secs']['soma'] = {'geom': {}, 'mechs': {}}  # soma params dict
        PYRcell['secs']['soma']['geom'] = {'diam': '19 + uniform(-0.5, 0.5)', 'L': 18.8, 'Ra': 123.0}  # soma geometry
        PYRcell['secs']['soma']['mechs']['hh'] = {'gnabar': '0.12 + 0.001*ynorm', 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  # soma hh mechanism
        self.netParams.cellParams['PYR'] = PYRcell

        ## Population parameters
        self.netParams.popParams['S'] = {'cellType': 'PYR', 'numCells': 20}
        self.netParams.popParams['M'] = {'cellType': 'PYR', 'numCells': 20}

        ## Synaptic mechanism parameters
        self.netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}  # excitatory synaptic mechanism

        ## Cell connectivity rules
        self.netParams.connParams['S->M'] = {    #  S -> M label
            'preConds': {'pop': 'S'},       # conditions of presyn cells
            'postConds': {'pop': 'M'},      # conditions of postsyn cells
            'probability': 0.5,
            'weight': 0.01,                 # synaptic weight
            'delay': 5, 
            'synMech': 'exc'}               # synaptic mechanism


    def add_evoked_drive(self):
    # def add_evoked_drive(self, name, *, mu, sigma, numspikes, location,
    #                      n_drive_cells='n_cells', cell_specific=True,
    #                      weights_ampa=None, weights_nmda=None,
    #                      space_constant=3., synaptic_delays=0.1,
    #                      probability=1.0, event_seed=2, conn_seed=3):
        """Add an 'evoked' external drive to the network

        # create drive cells and connect them to the real cells defined in the init function

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
        location : str
            Target location of synapses. Must be an element of
            `Cell.sect_loc` such as 'proximal' or 'distal', which defines a
            group of sections, or an existing section such as 'soma' or
            'apical_tuft' (defined in `Cell.sections` for all targeted cells).
            The parameter `legacy_mode` of the `Network` must be set to `False`
            to target specific sections.
        n_drive_cells : int | 'n_cells'
            The number of drive cells that each contribute an independently
            sampled synaptic spike to the network according to the Gaussian
            time distribution (mu, sigma). If n_drive_cells='n_cells'
            (default) and cell_specific=True, a drive cell gets assigned to
            each available simulated cell in the network with 1-to-1
            connectivity. Otherwise, drive cells are assigned with
            all-to-all connectivity. If you wish to synchronize the timing of
            this evoked drive across the network in a given trial with one
            spike, set n_drive_cells=1 and cell_specific=False.
        cell_specific : bool
            Whether each artifical drive cell has 1-to-1 (True, default) or
            all-to-all (False) connection parameters. Note that 1-to-1
            connectivity requires that n_drive_cells='n_cells', where 'n_cells'
            denotes the number of all available cells that this drive can
            target in the network.
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from the column origin) of synaptic
            weights and delays within the simulated column. The constant is
            measured in the units of ``inplane_distance`` of
            :class:`~hnn_core.Network`. For example, for ``space_constant=3``,
            the weights are modulated by the factor
            ``exp(-(x / (3 * inplane_distance)) ** 2)``, where x is the
            physical distance (in um) between the connected cells in the xy
            plane (delays are modulated by the inverse of this factor).
        probability : dict or float (default: 1.0)
            Probability of connection between any src-target pair.
            Use dict to create probability->cell mapping. If float, applies to
            all target cell types
        event_seed : int
            Optional initial seed for random number generator (default: 2).
            Used to generate event times for drive cells.
            Not fixed across trials (see Notes)
        conn_seed : int
            Optional initial seed for random number generator (default: 3).
            Used to randomly remove connections when probablity < 1.0.
            Fixed across trials (see Notes)

        Notes
        -----
        Random seeding behavior across trials is different for event_seed
        and conn_seed (n_trials > 1 in simulate_dipole(..., n_trials):

        - event_seed
            Across trials, the random seed is incremented leading such that
            the exact spike times are different
        - conn_seed
            The random seed does not change across trials. This means for
            probability < 1.0, the random subset of gids targeted is the same.
        """

        # Stimulation parameters
        self.netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 10, 'noise': 0.5}
        self.netParams.stimTargetParams['bkg->PYR'] = {'source': 'bkg', 'conds': {'cellType': 'PYR'}, 'weight': 0.01, 'delay': 1, 'synMech': 'exc'}

        
        

def netpyne_model(params):
	net = NetPyne_Model(params)
	return net

def simulate_dipole_netpyne(net,  dt=0.025, tstop=170):
	# code to simulate netpyne model

  # Simulation options
  simConfig = specs.SimConfig()       # object of class SimConfig to store simulation configuration

  simConfig.duration = tstop          # Duration of the simulation, in ms
  simConfig.dt = dt                # Internal integration timestep to use
  simConfig.verbose = False           # Show detailed messages
  simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
  simConfig.recordStep = 0.1          # Step size in ms to save data (eg. V traces, LFP, etc)
  simConfig.filename = 'tutxxx'  # Set file output name
  simConfig.savePickle = False        # Save params, network and sim output to pickle file
  simConfig.saveJson = True

  simConfig.analysis['plotRaster'] = {'saveFig': True}                  # Plot a raster
  simConfig.analysis['plotTraces'] = {'include': [1,2,3,4,5], 'saveFig': True}  # Plot recorded traces for this list of cells
  simConfig.analysis['plot2Dnet'] = {'saveFig': True}                   # plot 2D cell positions and connections

  sim.createSimulateAnalyze(netParams = net.netParams, simConfig = simConfig)

#   return ...






