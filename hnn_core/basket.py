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
        self.sect_loc = dict(proximal=['soma'], distal=[])

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
        # this is a pretty fast NMDA, no?
        self.synapses['soma_nmda'] = self.syn_create(
            self.soma(0.5), e=0., tau1=1., tau2=20.)

    # this function might make more sense as a method of net?
    # par: receive from external inputs
    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        # for some gid relating to the input feed:
        for gid_src, p_src, pos in zip(gid_dict['common'],
                                       p_ext, pos_dict['common']):
            # check if AMPA params are defined in the p_src
            if f'{self.name}_ampa' in p_src.keys():
                # create an nc_dict
                nc_dict = {
                    'pos_src': pos,
                    'A_weight': p_src[f'{self.name}_ampa'][0],
                    'A_delay': p_src[f'{self.name}_ampa'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                self._connect_feed_at_loc(
                    feed_loc='proximal', receptor='ampa',
                    gid_src=gid_src, nc_dict=nc_dict,
                    nc_list=self.ncfrom_common)

            # Check if NMDA params are defined in p_src
            if f'{self.name}_nmda' in p_src.keys():
                nc_dict = {
                    'pos_src': pos,
                    'A_weight': p_src[f'{self.name}_nmda'][0],
                    'A_delay': p_src[f'{self.name}_nmda'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                self._connect_feed_at_loc(
                    feed_loc='proximal', receptor='nmda',
                    gid_src=gid_src, nc_dict=nc_dict,
                    nc_list=self.ncfrom_common)

    # one parreceive function to handle all types of external parreceives
    # types must be defined explicitly here
    def parreceive_ext(self, type, gid, gid_dict, pos_dict, p_ext):
        """Recieve external inputs."""
        # shouldn't this just check for evprox?
        if type.startswith(('evprox', 'evdist')):
            if self.celltype in p_ext.keys():
                gid_ev = gid + gid_dict[type][0]

                nc_dict = dict()
                nc_dict['ampa'] = {
                    'pos_src': pos_dict[type][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                nc_dict['nmda'] = {
                    'pos_src': pos_dict[type][gid],
                    # index 1 is nmda weight
                    'A_weight': p_ext[self.celltype][1],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                # NEW: note that default/original is 0 nmda weight
                # for the soma (both prox and distal evoked)
                for receptor in ['ampa', 'nmda']:
                    self._connect_feed_at_loc(
                        feed_loc='proximal', receptor=receptor,
                        gid_src=gid_ev, nc_dict=nc_dict[receptor],
                        nc_list=self.ncfrom_ev)

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids
            # of the extgauss
            # pos_dict is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)
            if self.celltype in p_ext.keys():
                gid_extgauss = gid + gid_dict['extgauss'][0]

                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self._connect_feed_at_loc(
                    feed_loc='proximal', receptor='ampa',
                    gid_src=gid_extgauss, nc_dict=nc_dict,
                    nc_list=self.ncfrom_extgauss)

        elif type == 'extpois':
            if self.celltype in p_ext.keys():
                gid_extpois = gid + gid_dict['extpois'][0]

                nc_dict = {
                    'pos_src': pos_dict['extpois'][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self._connect_feed_at_loc(
                    feed_loc='proximal', receptor='ampa',
                    gid_src=gid_extpois, nc_dict=nc_dict,
                    nc_list=self.ncfrom_extpois)

                if p_ext[self.celltype][1] > 0.0:
                    # index 1 for nmda weight
                    nc_dict['A_weight'] = p_ext[self.celltype][1]
                    self._connect_feed_at_loc(
                        feed_loc='proximal', receptor='nmda',
                        gid_src=gid_extpois, nc_dict=nc_dict,
                        nc_list=self.ncfrom_extpois)

        else:
            print("Warning, type def not specified in L2Basket")


class L2Basket(BasketSingle):
    """Class for layer 2 basket cells."""

    def __init__(self, gid=-1, pos=-1):
        # BasketSingle.__init__(self, pos, L, diam, Ra, cm)
        # Note: Basket cell properties set in BasketSingle())
        BasketSingle.__init__(self, gid, pos, 'L2Basket')
        self.celltype = 'L2_basket'

        self._synapse_create()
        self._biophysics()

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
