"""Test dipole contribution analysis and visualization."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from hnn_core import Network, simulate_dipole, jones_2009_model
from hnn_core.dipole_contributions import (
    plot_dipole_contributions,
    analyze_dipole_contributions,
    _identify_up_down_components,
    _estimate_drive_contribution,
)


def test_identify_up_down_components():
    """Test identification of UP and DOWN components."""
    # Create a simple network with drives
    net = Network(add_drives_from_params=False)
    
    # Add proximal drive (should be UP component)
    net.add_evoked_drive(
        name='proximal_drive',
        mu=50.0,
        sigma=5.0,
        numspikes=1,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    # Add distal drive (should be DOWN component)
    net.add_evoked_drive(
        name='distal_drive',
        mu=100.0,
        sigma=5.0,
        numspikes=1,
        location='distal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    # Add tonic bias
    net.add_tonic_bias(
        amplitude={'L2_pyramidal': 0.5},
        t0=0.0,
        tstop=200.0
    )
    
    # Simulate dipole
    dpls = simulate_dipole(net, tstop=200.0, n_trials=1)
    dpl = dpls[0]
    
    # Test component identification
    up_components, down_components = _identify_up_down_components(net, dpl)
    
    # Check that we found the expected components
    assert len(up_components) > 0, "Should find UP components"
    assert len(down_components) > 0, "Should find DOWN components"
    
    # Check for expected component names
    up_names = list(up_components.keys())
    down_names = list(down_components.keys())
    
    # Should have L2 in UP components
    assert any('L2' in name for name in up_names), "L2 should be in UP components"
    
    # Should have proximal drive in UP components
    assert any('proximal' in name for name in up_names), "Proximal drive should be in UP"
    
    # Should have distal drive in DOWN components  
    assert any('distal' in name for name in down_names), "Distal drive should be in DOWN"
    
    # Check that contributions have correct length
    for contrib in up_components.values():
        assert len(contrib) == len(dpl.times), "Contribution should match dipole length"
    
    for contrib in down_components.values():
        assert len(contrib) == len(dpl.times), "Contribution should match dipole length"


def test_estimate_drive_contribution():
    """Test drive contribution estimation."""
    times = np.linspace(0, 200, 1000)
    
    # Test evoked drive
    drive_info = {
        'type': 'evoked',
        'dynamics': {'mu': 50.0, 'sigma': 5.0, 'numspikes': 1}
    }
    
    contrib_up = _estimate_drive_contribution(drive_info, times, polarity='up')
    contrib_down = _estimate_drive_contribution(drive_info, times, polarity='down')
    
    assert len(contrib_up) == len(times), "Contribution should match time length"
    assert len(contrib_down) == len(times), "Contribution should match time length"
    assert np.max(contrib_up) > 0, "UP contribution should be positive"
    assert np.min(contrib_down) < 0, "DOWN contribution should be negative"
    
    # Test poisson drive
    drive_info = {
        'type': 'poisson',
        'dynamics': {'tstart': 25.0, 'tstop': 150.0, 'rate_constant': 10.0}
    }
    
    contrib = _estimate_drive_contribution(drive_info, times, polarity='up')
    
    # Should be non-zero during drive period
    start_idx = np.argmin(np.abs(times - 25.0))
    end_idx = np.argmin(np.abs(times - 150.0))
    
    assert np.any(contrib[start_idx:end_idx] != 0), "Should have contribution during drive"
    assert np.all(contrib[:start_idx] == 0), "Should be zero before drive"


def test_analyze_dipole_contributions():
    """Test dipole contribution analysis."""
    # Use Jones 2009 model for realistic test
    net = jones_2009_model()
    
    # Simulate dipole
    dpls = simulate_dipole(net, tstop=170.0, n_trials=1)
    dpl = dpls[0]
    
    # Test analysis function
    analysis = analyze_dipole_contributions(net, dpl, verbose=False)
    
    # Check return structure
    assert isinstance(analysis, dict), "Should return dictionary"
    assert 'up_components' in analysis, "Should have up_components"
    assert 'down_components' in analysis, "Should have down_components"
    assert 'total_up_contribution' in analysis, "Should have total_up_contribution"
    assert 'total_down_contribution' in analysis, "Should have total_down_contribution"
    assert 'net_contribution' in analysis, "Should have net_contribution"
    
    # Check that contributions are arrays of correct length
    assert len(analysis['total_up_contribution']) == len(dpl.times)
    assert len(analysis['total_down_contribution']) == len(dpl.times)
    assert len(analysis['net_contribution']) == len(dpl.times)


def test_plot_dipole_contributions():
    """Test dipole contribution plotting."""
    # Create simple network
    net = Network(add_drives_from_params=False)
    net.add_evoked_drive(
        name='test_drive',
        mu=50.0,
        sigma=5.0,
        numspikes=1,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    # Simulate dipole
    dpls = simulate_dipole(net, tstop=100.0, n_trials=1)
    dpl = dpls[0]
    
    # Test plotting function
    fig = plot_dipole_contributions(net, dpl, show=False)
    
    assert fig is not None, "Should return figure object"
    assert len(fig.axes) > 0, "Should have axes"
    
    # Test with time limits
    fig2 = plot_dipole_contributions(
        net, dpl, tmin=20.0, tmax=80.0, show=False
    )
    assert fig2 is not None, "Should work with time limits"
    
    # Test with custom axes
    fig3, ax = plt.subplots()
    fig_returned = plot_dipole_contributions(net, dpl, ax=ax, show=False)
    assert fig_returned == fig3, "Should use provided axes"
    
    plt.close('all')  # Clean up figures


def test_dipole_methods():
    """Test Dipole object methods for contribution analysis."""
    # Create network and simulate
    net = Network(add_drives_from_params=False)
    net.add_evoked_drive(
        name='test_drive',
        mu=50.0,
        sigma=5.0,
        numspikes=1,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    dpls = simulate_dipole(net, tstop=100.0, n_trials=1)
    dpl = dpls[0]
    
    # Test plot_contributions method
    fig = dpl.plot_contributions(net, show=False)
    assert fig is not None, "plot_contributions should return figure"
    
    # Test analyze_contributions method
    analysis = dpl.analyze_contributions(net, verbose=False)
    assert isinstance(analysis, dict), "analyze_contributions should return dict"
    assert 'up_components' in analysis, "Should have up_components"
    assert 'down_components' in analysis, "Should have down_components"
    
    plt.close('all')  # Clean up figures


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with empty network
    net = Network(add_drives_from_params=False)
    dpls = simulate_dipole(net, tstop=50.0, n_trials=1)
    dpl = dpls[0]
    
    # Should still work with no drives
    up_components, down_components = _identify_up_down_components(net, dpl)
    
    # Should at least have layer components if dipole has L2/L5 data
    if 'L2' in dpl.data:
        assert 'L2_pyramidal' in up_components
    if 'L5' in dpl.data:
        assert 'L5_pyramidal' in down_components
    
    # Test plotting with no components
    fig = plot_dipole_contributions(net, dpl, show=False)
    assert fig is not None, "Should handle empty network"
    
    plt.close('all')


if __name__ == '__main__':
    test_identify_up_down_components()
    test_estimate_drive_contribution()
    test_analyze_dipole_contributions()
    test_plot_dipole_contributions()
    test_dipole_methods()
    test_edge_cases()
    print("All tests passed!")