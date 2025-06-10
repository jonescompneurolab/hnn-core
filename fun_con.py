import numpy as np
import matplotlib.pyplot as plt
from hnn_core import simulate_dipole, jones_2009_model, Network, JoblibBackend
from hnn_core.viz import plot_dipole, plot_connectivity_matrix, plot_cell_connectivity
from hnn_core.params import read_params
from hnn_core.cells_default import pyramidal, basket

def calc_pyramidal_coord(xxrange, yyrange, zdiff):
    """Create pyramidal cell coordinates."""
    from itertools import product
    return [pos for pos in product(xxrange, yyrange, [zdiff])]

def calc_basket_coord(n_pyr_x, n_pyr_y, zdiff, inplane_distance, weight):
    """Create basket cell coordinates."""
    from itertools import product
    xzero = np.arange(0, n_pyr_x, 3) * inplane_distance
    xone = np.arange(1, n_pyr_x, 3) * inplane_distance
    yeven = np.arange(0, n_pyr_y, 2) * inplane_distance
    yodd = np.arange(1, n_pyr_y, 2) * inplane_distance
    coords = [pos for pos in product(xzero, yeven)] + \
             [pos for pos in product(xone, yodd)]
    coords_sorted = sorted(coords, key=lambda pos: pos[1])
    return [(pos[0], pos[1], weight * zdiff) for pos in coords_sorted]

# Test 1: Create network with custom positions
print("Test 1: Creating network with custom positions...")
n_pyr_x, n_pyr_y = 8, 8  # smol grid for testing
inplane_distance = 1.5  
layer_separation = 1500.0  

xxrange = np.arange(n_pyr_x) * inplane_distance
yyrange = np.arange(n_pyr_y) * inplane_distance

# custom pos dict
layer_dict = {
    "L5_pyramidal": calc_pyramidal_coord(xxrange, yyrange, zdiff=0),
    "L2_pyramidal": calc_pyramidal_coord(xxrange, yyrange, zdiff=layer_separation),
    "L5_basket": calc_basket_coord(n_pyr_x, n_pyr_y, 0, inplane_distance, weight=0.2),
    "L2_basket": calc_basket_coord(n_pyr_x, n_pyr_y, layer_separation, 
                                  inplane_distance, weight=0.8),
    "origin": (xxrange[len(xxrange)//2], yyrange[len(yyrange)//2], 
               layer_separation/2),
}

params = read_params('hnn_core/param/default.json') 
custom_net = Network(params, custom_pos_dict=layer_dict)
print(f"✓ Network created with {len(custom_net.cell_types)} cell types")
print(f"  Cell counts: L2_pyr={len(layer_dict['L2_pyramidal'])}, "
      f"L5_pyr={len(layer_dict['L5_pyramidal'])}, "
      f"L2_bas={len(layer_dict['L2_basket'])}, "
      f"L5_bas={len(layer_dict['L5_basket'])}")

# Test 2: Add custom cell type
print("\nTest 2: Adding custom cell type...")

et_positions = layer_dict['L5_pyramidal'][0:10]
et_cell_template = pyramidal('L5Pyr', override_params={
    'L5Pyr_soma_gkbar_hh2': 0.02,  
    'L5Pyr_soma_gnabar_hh2': 0.20
})
custom_net.add_cell_type('L5_ET', et_cell_template, et_positions)
print(f"Added L5_ET cell type with {len(et_positions)} cells")

# Test 3: Add evoked drives (from the example)
print("\nTest 3: Adding evoked drives...")
# Distal drive
weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
custom_net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=274)

# Proximal drives
weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}
custom_net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=544)

weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
custom_net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=814)
print("Added 3 evoked drives")