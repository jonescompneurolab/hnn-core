"""Model for inhibitory cell class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

from .cell import _Cell

# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted


class BasketSingle(_Cell):
    """Inhibitory cell class.

    Attributes
    ----------
    synapses : dict
        The synapses that the cell can use for connections.
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    """

    def __init__(self, pos, cell_name='Basket', gid=None):
        self.props = self._get_soma_props(cell_name, pos)
        _Cell.__init__(self, self.props, gid=gid)
        # store cell name for later
        self.name = cell_name

        # Define 3D shape and position of cell. By default neuron uses xy plane
        # for height and xz plane for depth. This is opposite for model as a
        # whole, but convention is followed in this function ease use of gui.
        self.shape_soma()
        self.synapses = dict()

    def set_biophysics(self):
        self.soma.insert('hh2')

    def _get_soma_props(self, cell_name, pos):
        return {
            'pos': pos,
            'L': 39.,
            'diam': 20.,
            'cm': 0.85,
            'Ra': 200.,
            'name': cell_name,
        }

    def get_sections(self):
        """Get sections."""
        return [self.soma]

    # creation of synapses
    def _synapse_create(self):
        # creates synapses onto this cell
        self.synapses['soma_ampa'] = self.syn_create(
            self.soma(0.5), e=0., tau1=0.5, tau2=5.)
        self.synapses['soma_gabaa'] = self.syn_create(
            self.soma(0.5), e=-80, tau1=0.5, tau2=5.)
        self.synapses['soma_nmda'] = self.syn_create(
            self.soma(0.5), e=0., tau1=1., tau2=20.)


class L2Basket(BasketSingle):
    """Class for layer 2 basket cells.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.
    """

    def __init__(self, pos, gid=None):
        # BasketSingle.__init__(self, pos, L, diam, Ra, cm)
        # Note: Basket cell properties set in BasketSingle())
        BasketSingle.__init__(self, pos, cell_name='L2Basket', gid=gid)
        self.celltype = 'L2_basket'

        self._synapse_create()
        self.set_biophysics()
        self.sect_loc = dict(proximal=['soma'], distal=['soma'])


class L5Basket(BasketSingle):
    """Class for layer 5 basket cells.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.
    """

    def __init__(self, pos, gid=None):
        # Note: Cell properties are set in BasketSingle()
        BasketSingle.__init__(self, pos, cell_name='L5Basket', gid=gid)
        self.celltype = 'L5_basket'

        self._synapse_create()
        self.set_biophysics()
        self.sect_loc = dict(proximal=['soma'], distal=[])
