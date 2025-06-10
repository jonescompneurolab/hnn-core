import numpy as np
from hnn_core import Network
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

def rotate_positions(positions, axis, angle_deg):
    """Rotate list of (x, y, z) around a given axis by angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_mats = {
        'x': np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]]),
        'y': np.array([[c, 0, s],
                       [0, 1, 0],
                       [-s, 0, c]]),
        'z': np.array([[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]])
    }
    R = rot_mats[axis]
    return [tuple(R @ np.array(p)) for p in positions]

print('Starting EXAMPLE 1')
# Example 1: rotating around axis
n_pyr_x, n_pyr_y = 10, 10
inplane_distance = 1.0
layer_separation = 1307.4
xxrange = np.arange(n_pyr_x) * inplane_distance
yyrange = np.arange(n_pyr_y) * inplane_distance

layer_dict = {
    "L5_bottom": calc_pyramidal_coord(xxrange, yyrange, zdiff=0),
    "L2_bottom": calc_pyramidal_coord(xxrange, yyrange, zdiff=layer_separation),
    "L5_mid": calc_basket_coord(n_pyr_x, n_pyr_y, 0, inplane_distance, weight=0.2),
    "L2_mid": calc_basket_coord(n_pyr_x, n_pyr_y, layer_separation, inplane_distance, weight=0.8),
    "origin": (xxrange[len(xxrange)//2], yyrange[len(yyrange)//2], layer_separation/2),
}

rotations = [('original', None), ('x-90', ('x', 90)), ('y-90', ('y', 90)), ('z-90', ('z', 90))]
params = read_params('hnn_core/param/default.json') 

for name, rot in rotations:
    if rot:
        axis, angle = rot
        custom = {
            'L5_pyramidal': rotate_positions(layer_dict['L5_bottom'], axis, angle),
            'L2_pyramidal': rotate_positions(layer_dict['L2_bottom'], axis, angle),
            'L5_basket': rotate_positions(layer_dict['L5_mid'], axis, angle),
            'L2_basket': rotate_positions(layer_dict['L2_mid'], axis, angle),
            'origin': layer_dict['origin']
        }
    else:
        custom = {
            'L5_pyramidal': layer_dict['L5_bottom'],
            'L2_pyramidal': layer_dict['L2_bottom'],
            'L5_basket': layer_dict['L5_mid'],
            'L2_basket': layer_dict['L2_mid'],
            'origin': layer_dict['origin']
        }

    print(f'\nVisualizing: {name}')
    net = Network(params, custom_pos_dict=custom)
    #net.plot_cells()
    #print(net._custom_pos_dict)

print('Starting EXAMPLE 2')

et_positions = layer_dict['L5_bottom'][0:10]
et_cell_template = pyramidal('L5Pyr', override_params={'soma_gkbar_hh2': 0.02})
net = Network(params, custom_pos_dict=custom)
net.add_cell_type('L5_ET', et_cell_template, et_positions)
net.plot_cells()
