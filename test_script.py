import hnn_core
from hnn_core import  jones_2009_model, simulate_dipole, Network
import numpy as np
import matplotlib as plt

params = hnn_core.read_params('/home/mohamed/Desktop/PhD Thesis/auditory_evoked_simulation/HNN-AEF-main/HNN_Parameters/L_Contra.param')
net = jones_2009_model()


def martinotti_template():
    from hnn_core.cell import Section, Cell

    cell_name = 'martinotti'
    pos = (5, 5, 5)
    # pos = net.pos_dict['L5_basket'].copy()

    end_pts = [[0, 0, 0], [0, 0, 39.]]
    sections = {'soma': Section(L=39., diam=20., cm=0.85,
        Ra=200., end_pts=end_pts)}
    sections['soma'].syns = ['gabaa', 'nmda']
    
    synapses = {
        'gabaa': {
            'e': -80,
            'tau1': 0.5,
            'tau2': 5.
        },
        'nmda': {
            'e': 0,
            'tau1': 1.,
            'tau2': 20.
        }
    }
    sect_loc = dict(proximal=['soma'], distal=['soma'])

    return Cell(cell_name, pos,
            sections=sections,
            synapses=synapses,
            topology=None,
            sect_loc=sect_loc,
            gid=None)

x1= np.linspace(0, 7, 7)
y1= np.linspace(0, 5, 5)
xv, yv = np.meshgrid(x1, y1)
pos = list()
for (x, y) in zip(xv.ravel(), yv.ravel()):
    pos.append((x, y, 0.))

# net._add_cell_type('L5_martinotti', pos= pos, cell_template=martinotti_template())
# net.plot_cells()

weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=None,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=4)

simulate_dipole(net, tstop = 100, record_vsec=False)

# net.add_connection(src_gids='L5_martinotti', target_gids='L5_pyramidal', loc='apical_tuft', receptor='gabaa', weight= 0.025 , delay=1.0 ,lamtha=70.0 , allow_autapses= False, probability=1)

    