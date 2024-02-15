:orphan:

.. _network_file_content:

====================
Network File Content
====================

File Content Description
------------------------
hdf5 is the file format used for storing the Network object. The network is stored in a layered format. The first layer consists of the network attributes.
The attributes of the network are then broken down until the network can be representated as a collection of key value pairs. For example - cell_types is a network
attribute therefore in the first layer. The description of each cell type is in layer 2. Each cell has various sections. The description of a section is in layer 3.
Below are the attributes of the network.

object_type : str
    Type of object (Network) saved.
N_pyr_x : int
    Nr of cells (x).
N_pyr_y : int
    Nr of cells (y).
threshold : float
    Firing threshold of all cells.
celsius : float
    Temperature in degree celsius.
cell_types : dict of Cell Object
    key : name of cell type.
    value : Cell object.
gid_ranges : dict of dict
    key : cell name or drive name.
    value : dict.
pos_dict : dict
    key : cell type name.
    value : All co-ordintes of the cell types.
cell_response : Instance of Cell Response Object
    The Cell Response object.
external_drives : dict of dict
    key : external drive name.
    value : dict.
external_biases : dict of dict
    key : external bias name.
    value : dict.
connectivity : list of dict
    Contains connection info between cells and
    cells and external drives.
rec_arrays : dict of Extracellular Arrays
    key : extracellular array name.
    value : Instance of Extracellular Array object.
delay : float
    Synaptic delay in ms.

Cell Type Description
---------------------

cell_name : str
    Name of the cell.
pos : tuple
    The (x, y, z) coordinates.
sections : dict of Section
    Dictionary with keys as section name.
synapses : dict of dict
    Keys are name of synaptic mechanism. Each synaptic mechanism
    has keys for parameters of the mechanism, e.g., 'e', 'tau1',
    'tau2'.
cell_tree : dict of list
    Stores the tree representation of a cell.
sect_loc : dict of list
    Can have keys 'proximal' or 'distal' each containing
    names of section locations that are proximal or distal.
gid : int
    GID of the cell in a network (or None if not yet assigned).
dipole_pp : list of h.Dipole()
    The Dipole objects (see dipole.mod).
vsec : dict
    Recording of section specific voltage. Must be enabled
    by running simulate_dipole(net, record_vsec=True) or
    simulate_dipole(net, record_vsoma=True).
isec : dict
    Contains recording of section specific currents indexed
    by synapse type (keys can be soma_gabaa, soma_gabab etc.).
    Must be enabled by running simulate_dipole(net, record_isec=True)
    or simulate_dipole(net, record_isoma=True).
tonic_biases : list of h.IClamp
    The current clamps inserted at each section of the cell
    for tonic biasing inputs.

Section Description
-------------------

L : float
    length of a section in microns.
diam : float
    diameter of a section in microns.
cm : float
    membrane capacitance in micro-Farads.
Ra : float
    axial resistivity in ohm-cm.
end_pts : list of [x, y, z]
    The start and stop points of the section.
syns : list of str
    The synaptic mechanisms to add in this section.
mechs : dict
    Mechanisms to insert in this section. The keys
    are the names of the mechanisms and values
    are the properties.

Gid Range Description
---------------------

start : int
    Start of the gid_range.
stop : int
    End of the gid_range.

External Drive Description
--------------------------

name : str
    Unique name for the drive.
dynamics : dict
    Parameters describing how the temporal dynamics of spike trains in the
    drive. The keys are specific to the type of drive ('evoked', 'bursty',
    etc.).
location : str
    Target location of synapses.
cell_specific : bool
    Whether each artifical drive cell has 1-to-1 (True, default) or
    all-to-all (False) connection parameters.
weights_ampa : dict or None
    Synaptic weights (in uS) of AMPA receptors on each targeted cell
    type (dict keys).
weights_nmda : dict or None
    Synaptic weights (in uS) of NMDA receptors on each targeted cell
    type (dict keys).
probability : dict or float
    Probability of connection between any src-target pair.
synaptic_delays : dict or float
    Synaptic delay (in ms) at the column origin, dispersed laterally as
    a function of the space_constant. If float, applies to all target
    cell types.
event_seed : int
    Optional initial seed for random number generator.
conn_seed : int
    Optional initial seed for random number generator.
n_drive_cells : int | 'n_cells'
    The number of drive cells that each contribute an independently
    sampled synaptic spike to the network according to the Gaussian
    time distribution (mu, sigma).
events : list
    Contains the spike times of exogeneous inputs.
    
External Bias Description
-------------------------

cell_type : str
    The cell type whose cells will get the tonic input.
amplitude : float
    The amplitude of the input.
t0 : float
    The start time of tonic input (in ms).
tstop : float
    The end time of tonic input (in ms).

Connection Description
----------------------

target_types : str
    Cell type of target gids.
target_gids : list of int
    Identifer for targets of source cells.
num_targets : int
    Number of unique target gids.
src_type : str
    Cell type of source gids.
src_gids : list of int
    Identifier for source cells.
num_srcs : int
    Number of unique source gids.
gid_pairs : dict
    dict indexed by src gids.
loc : str
    Target location of synapses.
receptor : str
    Synaptic receptor of connection.
nc_dict : dict
    Contains information about delay, weight, lamtha etc.
allow_autapses : bool
    If True, allow connecting neuron to itself.
probability : float
    Probability of connection between any src-target pair.

Extracellular Array Description
-------------------------------

positions : tuple | list of tuple
    The (x, y, z) coordinates (in um) of the extracellular electrodes.
conductivity : float
    Extracellular conductivity, in S/m.
method : str
    Approximation to use.