import numpy as np
from hnn_core import simulate_dipole
from hnn_core.network import Network
from hnn_core.network_models import (
    _create_cell_coords, random_model, custom_cell_types_model,
    CellTypeBuilder
)
from hnn_core.cells_default import pyramidal, basket
from hnn_core.params import _short_name

def test_basic_random_model():
    """Test the new random model with L2_random instead of L2_basket."""
    print("=== Testing Random Model with L2_random ===")
    
    net = random_model(mesh_shape=(3, 3))
    
    # Validating cell counts
    print(f"Network created: {net}")
    print(f"Number of L2 pyramidal cells: {len(net.pos_dict['L2_pyramidal'])}")
    print(f"Number of L5 pyramidal cells: {len(net.pos_dict['L5_pyramidal'])}")
    print(f"Number of L2_random cells: {len(net.pos_dict['L2_random'])}")  # Our custom cell type!
    print(f"Number of L5 basket cells: {len(net.pos_dict['L5_basket'])}")
    
    # Verifying positions
    if 'L5_pyramidal' in net.pos_dict and len(net.pos_dict['L5_pyramidal']) > 0:
        l5_pyr_pos = net.pos_dict['L5_pyramidal'][0]
        l2_pyr_pos = net.pos_dict['L2_pyramidal'][0]
        print(f"\nSample L5 pyramidal position: {l5_pyr_pos}")
        print(f"Sample L2 pyramidal position: {l2_pyr_pos}")
        print(f"Layer separation: {l2_pyr_pos[2] - l5_pyr_pos[2]} um")
    
    # Add drive that includes our custom cell type
    weights_ampa = {
        'L2_pyramidal': 0.01, 
        'L5_pyramidal': 0.01,
        'L2_random': 0.005  # Our custom cell type can receive drives!
    }
    net.add_evoked_drive('evprox1', mu=40, sigma=5, numspikes=1,
                         weights_ampa=weights_ampa, location='proximal')
    
    dpls = simulate_dipole(net, tstop=100, dt=0.5, n_trials=1)
    print(f"\nSimulation completed. Peak dipole: {np.abs(dpls[0].data['agg']).max():.2f}")
    
    return net

def test_custom_cell_types_model():
    """Test the model with completely custom cell types."""
    print("\n=== Testing Custom Cell Types Model ===")
    
    net = custom_cell_types_model(mesh_shape=(3, 3))
    
    print(f"Network created with custom cell types:")
    for cell_type in net.cell_types:
        print(f"  {cell_type}: {len(net.gid_ranges[cell_type])} cells")
    
    # Test that our custom cells work with drives
    weights_ampa = {
        'L2_interneuron': 0.001,
        'L4_stellate': 0.002,
        'L2_pyramidal': 0.001,
        'L5_pyramidal': 0.001
    }
    net.add_evoked_drive('custom_drive', mu=50, sigma=5, numspikes=1,
                         location='proximal', weights_ampa=weights_ampa)
    
    dpls = simulate_dipole(net, tstop=100, dt=0.5, n_trials=1)
    print(f"Simulation completed. Peak dipole: {np.abs(dpls[0].data['agg']).max():.2f}")
    
    return net

def test_mixed_custom_positions():
    """Test creating network with custom positions and mixed cell types."""
    print("\n=== Testing Mixed Custom Positions ===")
    
    # Custom positions for our new cell types
    custom_positions = {
        'L5_pyramidal': [(0, 0, 0), (50, 0, 0), (0, 50, 0), (50, 50, 0)],
        'L2_pyramidal': [(0, 0, 1000), (50, 0, 1000), (0, 50, 1000), (50, 50, 1000)],
        'L2_random': [(25, 25, 800), (75, 25, 800)],  # Our custom cell type
        'L4_stellate': [(25, 25, 500), (75, 75, 500)],  # New cell type
        'origin': (25, 25, 500)
    }
    
    # Create cell types including our custom ones
    cell_types = {
        'L2_random': basket(cell_name='L2Random'),  # Using our flexible basket function
        'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
        'L4_stellate': CellTypeBuilder.create_stellate('L4_stellate'),
        'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal'))
    }
    
    # Minimal params needed for Network
    params = {
        'tstop': 200.0,
        'dt': 0.025,
        'celsius': 37.0,
        'threshold': 0.0,
        'N_trials': 1,
        't0_pois': 0.,
        'T_pois': -1,
        'dipole_smooth_win': 15.0,
        'dipole_scalefctr': 30e3,
        'prng_seedcore_input_prox': 2,
        'prng_seedcore_input_dist': 2,
        'prng_seedcore_extpois': 2,
        'prng_seedcore_extgauss': 2,
        'record_vsoma': 0,
        'record_isoma': 0,
        'record_vsec': 0,
        'record_isec': 0,
        'record_ca': 0,
        'save_spec_data': 0,
        'save_figs': 0,
        'save_dpl': 0,
    }
    
    net = Network(params, pos_dict=custom_positions, cell_types=cell_types)
    
    print(f"Custom network created with:")
    for cell_type, positions in net.pos_dict.items():
        if cell_type != 'origin':
            print(f"  {cell_type}: {len(positions)} cells")
    
    # Verify positions match
    for cell_type in custom_positions:
        if cell_type != 'origin':
            assert net.pos_dict[cell_type] == custom_positions[cell_type]
    print("Position verification passed!")
    
    return net


net1 = test_basic_random_model()
net2 = test_custom_cell_types_model()
net3 = test_mixed_custom_positions()



print("\n1. Random model with L2_random:")
net1.plot_cells()

print("\n2. Custom cell types model:")
net2.plot_cells()

print("\n3. Custom visualization with colors:")
net3.plot_cells()