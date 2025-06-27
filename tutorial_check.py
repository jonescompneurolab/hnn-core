'''
# test_modular_with_drives.py
from hnn_core.network_models import random_model
from hnn_core import simulate_dipole

# Create the network
net = random_model(mesh_shape=(3, 3))

# Add a simple evoked drive that targets our new cell type
weights_ampa = {
    'L2_pyramidal': 0.001,
    'L2_random': 0.0005,  # Our new cell type
}
net.add_evoked_drive(
    name='evprox1',
    mu=40,
    sigma=5,
    numspikes=1,
    location='proximal',
    weights_ampa=weights_ampa,
    weights_nmda=None
)

# Simulate
dpl = simulate_dipole(net, tstop=100.0, dt=0.025)
print("Simulation with drives completed successfully!")
'''

'''
import matplotlib.pyplot as plt

from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.network_models import add_erp_drives_to_jones_model

add_erp_drives_to_jones_model(net)

net.set_cell_positions(inplane_distance=30.)
'''

# test_custom_cells.py
from hnn_core.network_models import custom_cell_types_model
from hnn_core import simulate_dipole

# Create network with custom cell types
net = custom_cell_types_model(mesh_shape=(3, 3))

# Add drive that targets custom cells
weights_ampa = {
    'L2_interneuron': 0.001,
    'L4_stellate': 0.002,
    'L2_pyramidal': 0.001,
    'L5_pyramidal': 0.001
}
net.add_evoked_drive(
    name='custom_drive',
    mu=50,
    sigma=5,
    numspikes=1,
    location='proximal',
    weights_ampa=weights_ampa
)

# Simulate
dpl = simulate_dipole(net, tstop=100.0)
print(f"Network has cell types: {list(net.cell_types.keys())}")

# Plot with automatic colors/markers
net.plot_cells()

# Or provide custom colors and markers
custom_colors = {
    'L2_random': 'green',
    'L2_interneuron': 'orange',
    'L4_stellate': 'purple'
}

custom_markers = {
    'L2_random': 'd',  # diamond
    'L2_interneuron': 'h',  # hexagon
    'L4_stellate': 'p'  # pentagon
}



net.plot_cells(cell_colors=custom_colors, cell_markers=custom_markers)

from hnn_core.viz import plot_dipole
#net.cell_types['L4_stellate'].plot_morphology() #morphology needs to be defined for new cell types


