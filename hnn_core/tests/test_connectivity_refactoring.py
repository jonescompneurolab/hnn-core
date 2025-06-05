from hnn_core import jones_2009_model, Network
from hnn_core.params import read_params
import os.path as op
import hnn_core


def test_cell_types_refactoring():
    """Tests cell_types parameter works correctly in Network class"""
    print("Testing cell_types refactoring...")
    
    # Test 1: Network with cell_types passed (new way)
    from hnn_core.cells_default import pyramidal, basket
    from hnn_core.params import _short_name
    
    custom_cell_types = {
        'L2_basket': basket(cell_name=_short_name('L2_basket')),
        'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
        'L5_basket': basket(cell_name=_short_name('L5_basket')),
        'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal'))
    }
    
    # laoding params
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_file = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_file)
    
    #network with explicit cell_types
    net_new = Network(params, cell_types=custom_cell_types)
    print(f"***Network created with cell_types parameter: {list(net_new.cell_types.keys())}***")
    
    # Test 2: Network without cell_types mentioned(backward test)
    net_default = Network(params, cell_types=None)
    print(f"***Network created with default cell_types: {list(net_default.cell_types.keys())}***")
    
    # Test 3: Verify both produce same
    assert set(net_new.cell_types.keys()) == set(net_default.cell_types.keys())
    print("***Both methods produce identical cell types***")
    
    return True


def test_connectivity_dict_refactoring():
    """Test that connectivity dictionary produces correct connections"""
    print("\nTesting connectivity dictionary refactoring...")
    
    # refactored jones_2009_model
    net = jones_2009_model()
    
    # Count connections by type
    conn_counts = {}
    for conn in net.connectivity:
        key = (conn['src_type'], conn['target_type'], conn['receptor'])
        conn_counts[key] = conn_counts.get(key, 0) + 1
    
    # verifying some connections to be existing
    required_connections = [
        ('L2_pyramidal', 'L2_pyramidal', 'ampa'),
        ('L2_pyramidal', 'L2_pyramidal', 'nmda'),
        ('L2_basket', 'L2_pyramidal', 'gabaa'),
        ('L5_basket', 'L5_pyramidal', 'gabab'),
        ('L2_pyramidal', 'L5_pyramidal', 'ampa'),
    ]
    
    for src, tgt, rec in required_connections:
        key = (src, tgt, rec)
        assert key in conn_counts, f"Missing connection: {src} -> {tgt} ({rec})"
        print(f" Found {src} -> {tgt} ({rec}): {conn_counts[key]} connection(s)")
    
    # L2_pyr -> L5_pyr has 2 locations
    l2_to_l5_conns = [c for c in net.connectivity 
                      if c['src_type'] == 'L2_pyramidal' 
                      and c['target_type'] == 'L5_pyramidal']
    locations = [c['loc'] for c in l2_to_l5_conns]
    assert 'proximal' in locations and 'distal' in locations
    print(" L2_pyramidal -> L5_pyramidal has both proximal and distal connections")
    
    return True


def main():
    """Run all tests"""
    print("REFACTORING VERIFICATION TESTS")
    
    # Test 1
    test_cell_types_refactoring()
    
    # Test 2  
    test_connectivity_dict_refactoring()
    
    # Test 3
    print("\nTesting full model creation...")
    net = jones_2009_model()
    print(f" jones_2009_model created successfully")
    print(f"  - Cell types: {list(net.cell_types.keys())}")
    print(f"  - Connections: {len(net.connectivity)}")
    print(f"  - Positions defined: {list(net.pos_dict.keys())}")
    
    # quick sim test
    print("\nRunning quick simulation test...")
    from hnn_core import simulate_dipole
    dpl = simulate_dipole(net, tstop=5, dt=0.5, n_trials=1)
    print(" Simulation completed successfully")
    
    print("Phew...Refactoring verified!")
    print("="*60)


if __name__ == "__main__":
    main()