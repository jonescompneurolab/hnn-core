import numpy as np
from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.network import Network
from hnn_core.network_models import _create_cell_coords
from hnn_core.cells_default import pyramidal, basket
from hnn_core.params import _short_name

def test_basic_jones_model():
    """Test basic Jones 2009 model creation."""
    
    net = jones_2009_model(mesh_shape=(10, 10))
    
    # validating cell counts
    print(f"Network created: {net}")
    print(f"Number of L2 pyramidal cells: {len(net.pos_dict['L2_pyramidal'])}")
    print(f"Number of L5 pyramidal cells: {len(net.pos_dict['L5_pyramidal'])}")
    print(f"Number of L2 basket cells: {len(net.pos_dict['L2_basket'])}")
    print(f"Number of L5 basket cells: {len(net.pos_dict['L5_basket'])}")
    
    # verifying pos
    l5_pyr_pos = net.pos_dict['L5_pyramidal'][0]
    l2_pyr_pos = net.pos_dict['L2_pyramidal'][0]
    print(f"\nSample L5 pyramidal position: {l5_pyr_pos}")
    print(f"Sample L2 pyramidal position: {l2_pyr_pos}")
    print(f"Layer separation: {l2_pyr_pos[2] - l5_pyr_pos[2]} um")
    
    weights_ampa = {'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01}
    net.add_evoked_drive('evprox1', mu=40, sigma=5, numspikes=1,
                         weights_ampa=weights_ampa, location='proximal')
    
    dpls = simulate_dipole(net, tstop=100, dt=0.5, n_trials=1)
    print(f"\nSimulation completed. Peak dipole: {np.abs(dpls[0].data['agg']).max():.2f}")
    
    return net

def test_custom_positions():
    """Test creating network with custom positions."""
    print("\n=== Testing Custom Positions ===")
    
    # custom pos for a smaller network
    custom_positions = {
        'L5_pyramidal': [(0, 0, 0), (50, 0, 0), (0, 50, 0), (50, 50, 0)],
        'L2_pyramidal': [(0, 0, 1000), (50, 0, 1000), (0, 50, 1000), (50, 50, 1000)],
        'L5_basket': [(25, 25, 200)],
        'L2_basket': [(25, 25, 800)],
        'origin': (25, 25, 500)
    }
    
    cell_types = {
        'L2_basket': basket(cell_name=_short_name('L2_basket')),
        'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
        'L5_basket': basket(cell_name=_short_name('L5_basket')),
        'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal'))
    }
 
    params = {'threshold': 0.0, 'celsius': 37.0} 
    net = Network(params, pos_dict=custom_positions, cell_types=cell_types)
    
    print(f"Custom network created with:")
    for cell_type, positions in net.pos_dict.items():
        if cell_type != 'origin':
            print(f"  {cell_type}: {len(positions)} cells")
    
    # verify if positions match
    for cell_type in ['L5_pyramidal', 'L2_pyramidal']:
        assert net.pos_dict[cell_type] == custom_positions[cell_type]
    print("Position verification passed!")
    
    return net

def test_layer_based_positioning():
    """Test the modular layer-based positioning approach."""
    print("\n=== Testing Layer-Based Positioning ===")
    
    # layer pos
    layer_dict = _create_cell_coords(n_pyr_x=5, n_pyr_y=5, 
                                     zdiff=1200, inplane_distance=2.0)
    
    print("Available layers:")
    for layer_name, positions in layer_dict.items():
        if layer_name != 'origin':
            print(f"  {layer_name}: {len(positions)} positions, "
                  f"z-coord = {positions[0][2] if positions else 'N/A'}")
    
    pos_dict = {
        'L5_pyramidal': layer_dict['L5_bottom'],
        'L5_basket': layer_dict['L5_mid'],  
        'L2_pyramidal': layer_dict['L2_bottom'],
        'L2_basket': layer_dict['L2_mid'],
        'origin': layer_dict['origin']
    }
    
    cell_types = {
        'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal')),
        'L5_basket': basket(cell_name=_short_name('L5_basket')),
        'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
        'L2_basket': basket(cell_name=_short_name('L2_basket'))
    }
    
    params = {'threshold': 0.0, 'celsius': 37.0}
    net = Network(params, pos_dict=pos_dict, cell_types=cell_types)
    
    print(f"\nNetwork created with layer-based positioning")
    for cell_type in net.cell_types:
        print(f"  {cell_type}: {len(net.gid_ranges[cell_type])} cells")
    
    return net

def test_custom_cell_arrangement():
    """Test creating network with some non-std cell arrangements."""
    print("\n=== Testing Custom Cell Arrangement ===")
    
    # basket cells are placed differently
    layer_dict = _create_cell_coords(n_pyr_x=6, n_pyr_y=6, 
                                     zdiff=1300, inplane_distance=1.5)
    
    # basket cells at edges of layers
    l5_positions = np.array(layer_dict['L5_bottom'])
    l2_positions = np.array(layer_dict['L2_bottom'])
    
    # edge positions for basket cells (first and last rows)
    edge_indices = list(range(6)) + list(range(30, 36))  # First and last row
    
    pos_dict = {
        'L5_pyramidal': layer_dict['L5_bottom'],
        'L2_pyramidal': layer_dict['L2_bottom'],
        'L5_basket': [tuple(l5_positions[i] + [0, 0, 100]) for i in edge_indices[:6]],
        'L2_basket': [tuple(l2_positions[i] - [0, 0, 100]) for i in edge_indices[:6]],
        'origin': layer_dict['origin']
    }
    
    # cell types
    cell_types = {
        'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal')),
        'L5_basket': basket(cell_name=_short_name('L5_basket')),
        'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
        'L2_basket': basket(cell_name=_short_name('L2_basket'))
    }
    
    params = {'threshold': 0.0, 'celsius': 37.0}
    net = Network(params, pos_dict=pos_dict, cell_types=cell_types)
    
    print(f"Network with custom basket cell arrangement:")
    print(f"  Pyramidal cells arranged in 6x6 grid")
    print(f"  Basket cells placed at layer edges")
    for cell_type in net.cell_types:
        print(f"  {cell_type}: {len(net.gid_ranges[cell_type])} cells")
    
    return net

print("Testing!!!")
print("=" * 60)

# Test 1: Basic Jones model
net1 = test_basic_jones_model()

# Test 2: Custom positions
net2 = test_custom_positions()

# Test 3: Layer-based positioning
net3 = test_layer_based_positioning()

# Test 4: Custom cell arrangement
net4 = test_custom_cell_arrangement()

print("\n=== Visualizing Networks with plot_cells() ===")

net1.plot_cells()
net2.plot_cells()
net3.plot_cells()
net4.plot_cells()