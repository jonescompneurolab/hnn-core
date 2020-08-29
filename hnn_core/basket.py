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

    def __init__(self, gid, pos, cell_name='Basket'):
        self.props = self.__set_props(cell_name, pos)
        _Cell.__init__(self, gid, self.props)
        # store cell name for later
        self.name = cell_name

        # Define 3D shape and position of cell. By default neuron uses xy plane
        # for height and xz plane for depth. This is opposite for model as a
        # whole, but convention is followed in this function ease use of gui.
        self.shape_soma()
        self.synapses = dict()

    def _biophysics(self):
        self.soma.insert('hh2')

    def __set_props(self, cell_name, pos):
        return {
            'pos': pos,
            'L': 39.,
            'diam': 20.,
            'cm': 0.85,
            'Ra': 200.,
            'name': cell_name,
        }

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
    """Class for layer 2 basket cells."""

    def __init__(self, gid=-1, pos=-1):
        # BasketSingle.__init__(self, pos, L, diam, Ra, cm)
        # Note: Basket cell properties set in BasketSingle())
        BasketSingle.__init__(self, gid, pos, 'L2Basket')
        self.celltype = 'L2_basket'

        self._synapse_create()
        self._biophysics()
        self.sect_loc = dict(proximal=['soma'], distal=['soma'])

    # par connect between all presynaptic cells
    # no connections from L5Pyr or L5Basket to L2Baskets
    def parconnect(self, gid, gid_dict, pos_dict, p):
        self._connect(gid, gid_dict, pos_dict, p, 'L2_pyramidal', 'L2Pyr',
                      postsyns=[self.synapses['soma_ampa']])
        self._connect(gid, gid_dict, pos_dict, p, 'L2_basket', 'L2Basket',
                      lamtha=20., postsyns=[self.synapses['soma_gabaa']])


class L5Basket(BasketSingle):
    """Class for layer 5 basket cells."""

    def __init__(self, gid=-1, pos=-1):
        # Note: Cell properties are set in BasketSingle()
        BasketSingle.__init__(self, gid, pos, 'L5Basket')
        self.celltype = 'L5_basket'

        self._synapse_create()
        self._biophysics()
        self.sect_loc = dict(proximal=['soma'], distal=[])

    # connections FROM other cells TO this cell
    # there are no connections from the L2Basket cells. congrats!
    def parconnect(self, gid, gid_dict, pos_dict, p):
        self._connect(gid, gid_dict, pos_dict, p, 'L5_basket', 'L5Basket',
                      lamtha=20., autapses=False,
                      postsyns=[self.synapses['soma_gabaa']])
        self._connect(gid, gid_dict, pos_dict, p, 'L5_pyramidal', 'L5Pyr',
                      postsyns=[self.synapses['soma_ampa']])
        self._connect(gid, gid_dict, pos_dict, p, 'L2_pyramidal', 'L2Pyr',
                      postsyns=[self.synapses['soma_ampa']])
