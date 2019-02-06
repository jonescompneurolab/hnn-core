"""Model for inhibitory cell class."""

from .cell import Cell

# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted


class BasketSingle(Cell):
    """Inhibitory cell class."""

    def __init__(self, gid, pos, cell_name='Basket'):
        self.props = self.__set_props(cell_name, pos)
        Cell.__init__(self, gid, self.props)
        # store cell name for later
        self.name = cell_name
        # set 3D shape - unused for now but a prototype
        self.__shape_change()

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

    # Define 3D shape and position of cell. By default neuron uses xy plane for
    # height and xz plane for depth. This is opposite for model as a whole, but
    # convention is followed in this function ease use of gui.
    def __shape_change(self):
        self.shape_soma()

    # creation of synapses
    def _synapse_create(self):
        # creates synapses onto this cell
        self.soma_ampa = self.syn_ampa_create(self.soma(0.5))
        self.soma_gabaa = self.syn_gabaa_create(self.soma(0.5))
        self.soma_nmda = self.syn_nmda_create(self.soma(0.5))

    def _connect(self, gid, gid_dict, pos_dict, p, type_src, name_src,
                 lamtha=3., synapse='ampa'):
        for gid_src, pos in zip(gid_dict[type_src],
                                pos_dict[type_src]):
            if gid_src == gid:
                continue
            nc_dict = {
                'pos_src': pos,
                'A_weight': p['gbar_%s_%s' % (name_src, self.name)],
                'A_delay': 1.,
                'lamtha': lamtha,
                'threshold': p['threshold'],
                'type_src': type_src
            }

            getattr(self, 'ncfrom_%s' % name_src).append(
                self.parconnect_from_src(
                    gid_src, nc_dict, getattr(self, 'soma_%s' % synapse)))


class L2Basket(BasketSingle):
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
        self._connect(gid, gid_dict, pos_dict, p, 'L2_pyramidal', 'L2Pyr')
        self._connect(gid, gid_dict, pos_dict, p, 'L2_basket', 'L2Basket',
                      lamtha=20., synapse='gabaa')

    # this function might make more sense as a method of net?
    # par: receive from external inputs
    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        # for some gid relating to the input feed:
        for gid_src, p_src, pos in zip(gid_dict['extinput'], p_ext, pos_dict['extinput']):
            # check if AMPA params are defined in the p_src
            if 'L2Basket_ampa' in p_src.keys():
                # create an nc_dict
                nc_dict_ampa = {
                    'pos_src': pos,
                    'A_weight': p_src['L2Basket_ampa'][0],
                    'A_delay': p_src['L2Basket_ampa'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # AMPA synapse
                self.ncfrom_extinput.append(self.parconnect_from_src(
                    gid_src, nc_dict_ampa, self.soma_ampa))

            # Check if NMDA params are defined in p_src
            if 'L2Basket_nmda' in p_src.keys():
                nc_dict_nmda = {
                    'pos_src': pos,
                    'A_weight': p_src['L2Basket_nmda'][0],
                    'A_delay': p_src['L2Basket_nmda'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # NMDA synapse
                self.ncfrom_extinput.append(self.parconnect_from_src(
                    gid_src, nc_dict_nmda, self.soma_nmda))

    # one parreceive function to handle all types of external parreceives
    # types must be defined explicitly here
    def parreceive_ext(self, type, gid, gid_dict, pos_dict, p_ext):
        if type.startswith(('evprox', 'evdist')):
            if self.celltype in p_ext.keys():
                gid_ev = gid + gid_dict[type][0]

                nc_dict_ampa = {
                    'pos_src': pos_dict[type][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                nc_dict_nmda = {
                    'pos_src': pos_dict[type][gid],
                    # index 1 is nmda weight
                    'A_weight': p_ext[self.celltype][1],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                # connections depend on location of input - why only for L2 basket and not L5 basket?
                if p_ext['loc'] is 'proximal':
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_ampa, self.soma_ampa))
                    # NEW: note that default/original is 0 nmda weight for the soma (for prox evoked)
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_nmda, self.soma_nmda))

                elif p_ext['loc'] is 'distal':
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_ampa, self.soma_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_nmda, self.soma_nmda))

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids of the extgauss
            # pos_list is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)
            # I recognize this is ugly (hack)
            if self.celltype in p_ext.keys():
                # since gid ids are unique, then these will all be shifted.
                # if order of extgauss random feeds ever matters (likely)
                # then will have to preserve order
                # of creation based on gid ids of the cells
                # this is a dumb place to put this information
                gid_extgauss = gid + gid_dict['extgauss'][0]

                # gid works here because there are as many pos items in pos_dict['extgauss'] as there are cells
                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][1],  # index 2 is delay
                    'lamtha': p_ext['lamtha'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self.ncfrom_extgauss.append(self.parconnect_from_src(
                    gid_extgauss, nc_dict, self.soma_ampa))

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

                self.ncfrom_extpois.append(self.parconnect_from_src(
                    gid_extpois, nc_dict, self.soma_ampa))

                if p_ext[self.celltype][1] > 0.0:
                    # index 1 for nmda weight
                    nc_dict['A_weight'] = p_ext[self.celltype][1]
                    self.ncfrom_extpois.append(self.parconnect_from_src(
                        gid_extpois, nc_dict, self.soma_nmda))

        else:
            print("Warning, type def not specified in L2Basket")


class L5Basket(BasketSingle):
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
                      lamtha=20., synapse='gabaa')
        self._connect(gid, gid_dict, pos_dict, p, 'L5_pyramidal', 'L5Pyr')
        self._connect(gid, gid_dict, pos_dict, p, 'L2_pyramidal', 'L2Pyr')

    # parallel receive function parreceive()
    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        for gid_src, p_src, pos in zip(gid_dict['extinput'], p_ext,
                                       pos_dict['extinput']):
            # Check if AMPA params are define in p_src
            if 'L5Basket_ampa' in p_src.keys():
                nc_dict_ampa = {
                    'pos_src': pos,
                    'A_weight': p_src['L5Basket_ampa'][0],
                    'A_delay': p_src['L5Basket_ampa'][1],  # right index??
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # AMPA synapse
                self.ncfrom_extinput.append(self.parconnect_from_src(
                    gid_src, nc_dict_ampa, self.soma_ampa))

            # Check if nmda params are define in p_src
            if 'L5Basket_nmda' in p_src.keys():
                nc_dict_nmda = {
                    'pos_src': pos,
                    'A_weight': p_src['L5Basket_nmda'][0],
                    'A_delay': p_src['L5Basket_nmda'][1],  # right index??
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # NMDA synapse
                self.ncfrom_extinput.append(self.parconnect_from_src(
                    gid_src, nc_dict_nmda, self.soma_nmda))

    # one parreceive function to handle all types of external parreceives
    # types must be defined explicitly here
    def parreceive_ext(self, type, gid, gid_dict, pos_dict, p_ext):
        # shouldn't this just check for evprox?
        if type.startswith(('evprox', 'evdist')):
            if self.celltype in p_ext.keys():
                gid_ev = gid + gid_dict[type][0]

                nc_dict_ampa = {
                    'pos_src': pos_dict[type][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                nc_dict_nmda = {
                    'pos_src': pos_dict[type][gid],
                    # index 1 is nmda weight
                    'A_weight': p_ext[self.celltype][1],
                    'A_delay': p_ext[self.celltype][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self.ncfrom_ev.append(self.parconnect_from_src(
                    gid_ev, nc_dict_ampa, self.soma_ampa))

                # NEW: note that default/original is 0 nmda weight for the soma (both prox and distal evoked)
                self.ncfrom_ev.append(self.parconnect_from_src(
                    gid_ev, nc_dict_nmda, self.soma_nmda))

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids of the extgauss
            # pos_dict is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)
            if 'L5_basket' in p_ext.keys():
                gid_extgauss = gid + gid_dict['extgauss'][0]

                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    # index 0 is ampa weight
                    'A_weight': p_ext['L5_basket'][0],
                    'A_delay': p_ext['L5_basket'][2],  # index 2 is delay
                    'lamtha': p_ext['lamtha'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self.ncfrom_extgauss.append(self.parconnect_from_src(
                    gid_extgauss, nc_dict, self.soma_ampa))

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

                self.ncfrom_extpois.append(self.parconnect_from_src(
                    gid_extpois, nc_dict, self.soma_ampa))

                if p_ext[self.celltype][1] > 0.0:
                    # index 1 for nmda weight
                    nc_dict['A_weight'] = p_ext[self.celltype][1]
                    self.ncfrom_extpois.append(self.parconnect_from_src(
                        gid_extpois, nc_dict, self.soma_nmda))

        else:
            print("Warning, type def not specified in L2Basket")
