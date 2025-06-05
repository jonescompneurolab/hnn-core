import traceback

def test_custom_cell_types():
    print("trying custom cell types")
    try:
        from hnn_core import Network
        from hnn_core.cells_default import pyramidal, basket
        from hnn_core.params import _short_name
        
        # Create custom cell types using valid cell names
        custom_cells = {
            'L2_basket': basket(cell_name=_short_name('L2_basket')),
            'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal')),
        }
        
        net = Network(params={'threshold': 0.0}, cell_types=custom_cells, mesh_shape=(3, 3))
        
        assert len(net.cell_types) == 2, f"Expected 2 cell types, got {len(net.cell_types)}"
        assert 'L2_basket' in net.cell_types
        assert 'L5_pyramidal' in net.cell_types
        
        print("Custom cell types test passed")
        return True
    except Exception as e:
        print(f"Custom cell types test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that existing code still works"""
    print("Testing backward compatibility...")
    try:
        from hnn_core import jones_2009_model, simulate_dipole
        
        # Create network the old way
        net = jones_2009_model()
        assert len(net.cell_types) == 4, f"Expected 4 cell types, got {len(net.cell_types)}"
        assert 'L2_pyramidal' in net.cell_types
        assert 'L5_pyramidal' in net.cell_types
        
        # Quick sim to ensure it works
        dpl = simulate_dipole(net, tstop=10, n_trials=1)
        
        print("Backward compatibility test passed")
        return True
    except Exception as e:
        print(f"Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    
    tests = [
        test_custom_cell_types,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\n Some tests failed")
        return 1


if __name__ == "__main__":
    main()