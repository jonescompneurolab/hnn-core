"""The documentation functions."""

docdict = dict()
# Define docdicts

docdict[
    "net"
] = """
net : Instance of Network object
    The Network object.
"""

docdict[
    "fname"
] = """
fname : str | Path object
    Full path to the output file (.hdf5).
"""

docdict[
    "overwrite"
] = """
overwrite : Boolean
    True : Overwrite existing file.
    False : Throw error if file already exists.
"""

docdict[
    "save_unsimulated"
] = """
save_unsimulated : Boolean
    True : Do not save the Network simulation output.
    False : Save complete Network as provided in input.
"""

docdict[
    "read_raw"
] = """
read_raw : Boolean
    True : Read unsimulated network.
    False : Read simulated network.
"""

docdict[
    "L"
] = """
L : float
    length of a section in microns.
"""

docdict[
    "diam"
] = """
diam : float
    diameter of a section in microns.
"""

docdict[
    "cm"
] = """
cm : float
    membrane capacitance in micro-Farads.
"""

docdict[
    "Ra"
] = """
Ra : float
    axial resistivity in ohm-cm.
"""

docdict[
    "end_pts"
] = """
end_pts : list of [x, y, z]
    The start and stop points of the section.
"""

docdict[
    "syns"
] = """
syns : list of str
    The synaptic mechanisms to add in this section.
"""

docdict[
    "mechs"
] = """
mechs : dict
    Mechanisms to insert in this section. The keys
    are the names of the mechanisms and values
    are the properties.
"""

docdict[
    "cell_name"
] = """
cell_name : str
    Name of the cell.
"""

docdict[
    "pos"
] = """
pos : tuple
    The (x, y, z) coordinates.
"""

docdict[
    "sections"
] = """
sections : dict of Section
    Dictionary with keys as section name.
"""

docdict[
    "synapses"
] = """
synapses : dict of dict
    Keys are name of synaptic mechanism. Each synaptic mechanism
    has keys for parameters of the mechanism, e.g., 'e', 'tau1',
    'tau2'.
"""

docdict[
    "cell_tree"
] = """
cell_tree : dict of list
    Stores the tree representation of a cell.
"""

docdict[
    "sect_loc"
] = """
sect_loc : dict of list
    Can have keys 'proximal' or 'distal' each containing
    names of section locations that are proximal or distal.
"""

docdict[
    "gid"
] = """
gid : int
    GID of the cell in a network (or None if not yet assigned).
"""

docdict[
    "dipole_pp"
] = """
dipole_pp : list of h.Dipole()
    The Dipole objects (see dipole.mod).
"""

docdict[
    "vsec"
] = """
vsec : dict
    Recording of section specific voltage. Must be enabled
    by running simulate_dipole(net, record_vsec=True) or
    simulate_dipole(net, record_vsoma=True).
"""

docdict[
    "isec"
] = """
isec : dict
    Contains recording of section specific currents indexed
    by synapse type (keys can be soma_gabaa, soma_gabab etc.).
    Must be enabled by running simulate_dipole(net, record_isec=True)
    or simulate_dipole(net, record_isoma=True).
"""

docdict[
    "isec"
] = """
isec : dict
    Contains recording of section specific currents indexed
    by synapse type (keys can be soma_gabaa, soma_gabab etc.).
    Must be enabled by running simulate_dipole(net, record_isec=True)
    or simulate_dipole(net, record_isoma=True).
"""

docdict[
    "tonic_biases"
] = """
tonic_biases : list of h.IClamp
    The current clamps inserted at each section of the cell
    for tonic biasing inputs.
"""

docdict[
    "start"
] = """
start : int
    Start of the gid_range.
"""

docdict[
    "stop"
] = """
stop : int
    End of the gid_range.
"""

docdict[
    "drive_name"
] = """
name : str
    Unique name for the drive.
"""

docdict[
    "dynamics"
] = """
dynamics : dict
    Parameters describing how the temporal dynamics of spike trains in the
    drive. The keys are specific to the type of drive ('evoked', 'bursty',
    etc.).
"""

docdict[
    "location"
] = """
location : str
    Target location of synapses.
"""

docdict[
    "cell_specific"
] = """
cell_specific : bool
    Whether each artifical drive cell has 1-to-1 (True, default) or
    all-to-all (False) connection parameters.
"""

docdict[
    "weights_ampa"
] = """
weights_ampa : dict or None
    Synaptic weights (in uS) of AMPA receptors on each targeted cell
    type (dict keys).
"""

docdict[
    "weights_nmda"
] = """
weights_nmda : dict or None
    Synaptic weights (in uS) of NMDA receptors on each targeted cell
    type (dict keys).
"""

docdict[
    "probability"
] = """
probability : dict or float
    Probability of connection between any src-target pair.
    Use dict to create probability->cell mapping. If float, applies to
    all target cell types.
"""

docdict[
    "synaptic_delays"
] = """
synaptic_delays : dict or float
    Synaptic delay (in ms) at the column origin, dispersed laterally as
    a function of the space_constant. If float, applies to all target
    cell types.
"""

docdict[
    "event_seed"
] = """
event_seed : int
    Optional initial seed for random number generator.
"""

docdict[
    "conn_seed"
] = """
conn_seed : int
    Optional initial seed for random number generator.
"""

docdict[
    "n_drive_cells"
] = """
n_drive_cells : int | 'n_cells'
    The number of drive cells that each contribute an independently
    sampled synaptic spike to the network according to the Gaussian
    time distribution (mu, sigma).
"""

docdict[
    "events"
] = """
events : list
    Contains the spike times of exogeneous inputs.
"""

docdict[
    "cell_type"
] = """
cell_type : str
    The cell type whose cells will get the tonic input.
"""

docdict[
    "amplitude"
] = """
amplitude : float
    The amplitude of the input.
"""

docdict[
    "t0"
] = """
t0 : float
    The start time of tonic input (in ms).
"""

docdict[
    "tstop"
] = """
tstop : float
    The end time of tonic input (in ms).
"""

docdict[
    "target_types"
] = """
target_types : str
    Cell type of target gids.
"""

docdict[
    "target_gids"
] = """
target_gids : list of int
    Identifer for targets of source cells.
"""

docdict[
    "num_targets"
] = """
num_targets : int
    Number of unique target gids.
"""

docdict[
    "src_type"
] = """
src_type : str
    Cell type of source gids.
"""

docdict[
    "src_gids"
] = """
src_gids : list of int
    Identifier for source cells.
"""

docdict[
    "num_srcs"
] = """
num_srcs : int
    Number of unique source gids.
"""

docdict[
    "gid_pairs"
] = """
gid_pairs : dict
    dict indexed by src gids.
"""

docdict[
    "loc"
] = """
loc : str
    Target location of synapses.
"""

docdict[
    "receptor"
] = """
receptor : str
    Synaptic receptor of connection.
"""

docdict[
    "nc_dict"
] = """
nc_dict : dict
    Contains information about delay, weight, lamtha etc.
"""

docdict[
    "allow_autapses"
] = """
allow_autapses : bool
    If True, allow connecting neuron to itself.
"""

docdict[
    "connection_probability"
] = """
probability : float
    Probability of connection between any src-target pair.
"""

docdict[
    "positions"
] = """
positions : tuple | list of tuple
    The (x, y, z) coordinates (in um) of the extracellular electrodes.
"""

docdict[
    "conductivity"
] = """
conductivity : float
    Extracellular conductivity, in S/m.
"""

docdict[
    "method"
] = """
method : str
    Approximation to use.
"""

docdict[
    "min_distance"
] = """
min_distance : float
    To avoid numerical errors in calculating potentials, apply a minimum
    distance limit between the electrode contacts and the active neuronal
    membrane elements that act as sources of current.
"""

docdict[
    "times"
] = """
times : array-like, shape (n_times,) | None
    Optionally, provide precomputed voltage sampling times for electrodes
    at `positions`.
"""

docdict[
    "voltages"
] = """
voltages : array-like, shape (n_trials, n_electrodes, n_times) | None
    Optionally, provide precomputed voltages for electrodes at
    ``positions``.
"""

docdict[
    "object_type"
] = """
object_type : str
    Type of object (Network) saved.
"""

docdict[
    "N_pyr_x"
] = """
N_pyr_x : int
    Nr of cells (x).
"""

docdict[
    "N_pyr_y"
] = """
N_pyr_y : int
    Nr of cells (y).
"""

docdict[
    "threshold"
] = """
threshold : float
    Firing threshold of all cells.
"""

docdict[
    "celsius"
] = """
celsius : float
    Temperature in degree celsius.
"""

docdict[
    "cell_types"
] = """
cell_types : dict of Cell Object
    key : name of cell type.
    value : Cell object.
"""

docdict[
    "gid_ranges"
] = """
gid_ranges : dict of dict
    key : cell name or drive name.
    value : dict.
"""

docdict[
    "pos_dict"
] = """
pos_dict : dict
    key : cell type name.
    value : All co-ordintes of the cell types.
"""

docdict[
    "cell_response"
] = """
cell_response : Instance of Cell Response Object
    The Cell Response object.
"""

docdict[
    "external_drives"
] = """
external_drives : dict of dict
    key : external drive name.
    value : dict.
"""

docdict[
    "external_biases"
] = """
external_biases : dict of dict
    key : external bias name.
    value : dict.
"""

docdict[
    "connectivity"
] = """
connectivity : list of dict
    Contains connection info between cells and
    cells and external drives.
"""

docdict[
    "rec_arrays"
] = """
rec_arrays : dict of Extracellular Arrays
    key : extracellular array name.
    value : Instance of Extracellular Array object.
"""

docdict[
    "delay"
] = """
delay : float
    Synaptic delay in ms.
"""
