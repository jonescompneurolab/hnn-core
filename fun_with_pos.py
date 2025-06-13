from hnn_core import jones_2009_model
from hnn_core.network_models import add_erp_drives_to_jones_model
import numpy as np

# Test 1, seeing for basic functionality..does the OG work?
net1 = jones_2009_model()
net1.plot_cells()

# Test 2, seeing for custom position
custom_pos = {
    'L2_pyramidal': [(4.5, 4.5, 1307.4)],
    'L5_pyramidal': [(4.5, 4.5, 0.0)]
}
net2 = jones_2009_model(custom_positions=custom_pos)
print(f"Custom network created with {len(net2.pos_dict['L2_pyramidal'])} L2 pyramidal cells")
net2.plot_cells()

# Adding at weird extreme positions
custom_pos = {
    'L2_pyramidal': [(0, 0, 1307.4), (100, 0, 1307.4)],
    'L5_pyramidal': [(0, 0, 0), (100, 0, 0)]
}
net2_alt = jones_2009_model(custom_positions=custom_pos)
print(f"Custom network created with {len(net2_alt.pos_dict['L2_pyramidal'])} L2 pyramidal cells")
net2_alt.plot_cells()

# Making a circular pattern
n_cells = 16
radius = 200

l2_positions = []
l5_positions = []

for i in range(n_cells):
    angle = 2 * np.pi * i / n_cells
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    
    l2_positions.append((x, y, 1307.4))
    l5_positions.append((x, y, 0))

net3 = jones_2009_model(
    custom_positions={
        'L2_pyramidal': l2_positions,
        'L5_pyramidal': l5_positions
    }
)
print(f"Circular network created with {len(l2_positions)} cells per layer")
net3.plot_cells()

# HNN-ception
import numpy as np
from hnn_core import jones_2009_model

def letter_to_positions(letter, x_offset, y=0, z=0, spacing=40):
    grid_map = {
        'H': [
            "X   X",
            "X   X",
            "XXXXX",
            "X   X",
            "X   X"
        ],
        'N': [
            "X   X",
            "XX  X",
            "X X X",
            "X  XX",
            "X   X"
        ]
    }

    positions = []
    rows = grid_map[letter]
    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            if char == 'X':
                x = x_offset + col_idx * spacing
                y_pos = y - row_idx * spacing
                positions.append((x, y_pos, z))
    return positions

z_L2 = 1307.4
z_L5 = 0

l2_pyramidal = letter_to_positions('H', x_offset=-300, z=z_L2)
l2_basket = letter_to_positions('N', x_offset=-80, z=z_L2)
l5_basket = letter_to_positions('N', x_offset=140, z=z_L5)

net_hnn = jones_2009_model(custom_positions={
    'L2_pyramidal': l2_pyramidal,
    'L2_basket': l2_basket,
    'L5_basket': l5_basket
})

print("phewww")
net_hnn.plot_cells()

