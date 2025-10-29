"""Test custom loss functions in optimization."""

import numpy as np
import pytest

from hnn_core import Network, simulate_dipole
from hnn_core.optimization import (
    Optimizer,
    _spectral_power_loss,
    _phase_coherence_loss,
    _custom_rmse_with_weights,
)


def test_custom_objective_function_validation():
    """Test validation of custom objective function signatures."""
    net = Network(add_drives_from_params=False)
    constraints = {'param1': (0, 1)}
    
    def set_params_dummy(net, params):
        pass
    
    # Valid custom function
    def valid_custom_obj(
        initial_net, initial_params, set_params, predicted_params,
        update_params, obj_values, tstop, obj_fun_kwargs
    ):
        return 1.0
    
    # Should not raise error
    optimizer = Optimizer(
        initial_net=net,
        tstop=100,
        constraints=constraints,
        set_params=set_params_dummy,
        obj_fun=valid_custom_obj,
    )
    assert optimizer.obj_fun_name == "custom"
    
    # Invalid custom function - wrong number of parameters
    def invalid_custom_obj(param1, param2):
        return 1.0
    
    with pytest.raises(ValueError, match="must have exactly 8 parameters"):
        Optimizer(
            initial_net=net,
            tstop=100,
            constraints=constraints,
            set_params=set_params_dummy,
            obj_fun=invalid_custom_obj,
        )
    
    # Invalid custom function - wrong parameter names
    def invalid_custom_obj2(
        wrong_name, initial_params, set_params, predicted_params,
        update_params, obj_values, tstop, obj_fun_kwargs
    ):
        return 1.0
    
    with pytest.raises(ValueError, match="Parameter 1 should be 'initial_net'"):
        Optimizer(
            initial_net=net,
            tstop=100,
            constraints=constraints,
            set_params=set_params_dummy,
            obj_fun=invalid_custom_obj2,
        )


def test_spectral_power_loss():
    """Test spectral power loss function."""
    # Create a simple network
    net = Network(add_drives_from_params=False)
    net.add_poisson_drive(
        name='test_drive',
        tstart=10.0,
        tstop=90.0,
        rate_constant=10.0,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    def set_params_dummy(net, params):
        pass
    
    def update_params_dummy(initial_params, predicted_params):
        return {'param1': predicted_params[0]}
    
    # Create mock target PSD
    target_freqs = np.linspace(1, 50, 50)
    target_psd = np.ones_like(target_freqs)
    
    obj_values = []
    
    # Test the function
    obj = _spectral_power_loss(
        initial_net=net,
        initial_params={'param1': 0.5},
        set_params=set_params_dummy,
        predicted_params=[0.5],
        update_params=update_params_dummy,
        obj_values=obj_values,
        tstop=100.0,
        obj_fun_kwargs={
            'target_psd': target_psd,
            'target_freqs': target_freqs,
            'freq_range': (5, 20),
            'n_trials': 1,
        }
    )
    
    assert isinstance(obj, float)
    assert len(obj_values) == 1
    assert obj_values[0] == obj


def test_phase_coherence_loss():
    """Test phase coherence loss function."""
    # Create a simple network
    net = Network(add_drives_from_params=False)
    net.add_poisson_drive(
        name='test_drive',
        tstart=10.0,
        tstop=90.0,
        rate_constant=10.0,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    def set_params_dummy(net, params):
        pass
    
    def update_params_dummy(initial_params, predicted_params):
        return {'param1': predicted_params[0]}
    
    obj_values = []
    
    # Test the function
    obj = _phase_coherence_loss(
        initial_net=net,
        initial_params={'param1': 0.5},
        set_params=set_params_dummy,
        predicted_params=[0.5],
        update_params=update_params_dummy,
        obj_values=obj_values,
        tstop=100.0,
        obj_fun_kwargs={
            'target_coherence': 0.8,
            'freq_band': (8, 12),
            'n_trials': 1,  # Will use signal splitting
        }
    )
    
    assert isinstance(obj, float)
    assert len(obj_values) == 1
    assert obj_values[0] == obj


def test_custom_rmse_with_weights():
    """Test custom weighted RMSE loss function."""
    # Create a simple network and simulate target
    net = Network(add_drives_from_params=False)
    net.add_poisson_drive(
        name='test_drive',
        tstart=10.0,
        tstop=90.0,
        rate_constant=10.0,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01},
        weights_nmda={'L2_pyramidal': 0.01},
    )
    
    # Create target dipole
    target_dpls = simulate_dipole(net, tstop=100.0, n_trials=1)
    target_dpl = target_dpls[0]
    
    # Create time weights
    time_weights = np.ones(len(target_dpl.data['agg']))
    
    def set_params_dummy(net, params):
        pass
    
    def update_params_dummy(initial_params, predicted_params):
        return {'param1': predicted_params[0]}
    
    obj_values = []
    
    # Test the function
    obj = _custom_rmse_with_weights(
        initial_net=net,
        initial_params={'param1': 0.5},
        set_params=set_params_dummy,
        predicted_params=[0.5],
        update_params=update_params_dummy,
        obj_values=obj_values,
        tstop=100.0,
        obj_fun_kwargs={
            'target': target_dpl,
            'time_weights': time_weights,
            'n_trials': 1,
        }
    )
    
    assert isinstance(obj, float)
    assert len(obj_values) == 1
    assert obj_values[0] == obj


def test_optimizer_with_custom_loss():
    """Test Optimizer class with custom loss functions."""
    # Create a simple network
    net = Network(add_drives_from_params=False)
    
    def set_params_dummy(net, params):
        # Add a simple drive for testing
        net.external_drives.clear()
        net.connectivity = []
        net.add_poisson_drive(
            name='test_drive',
            tstart=10.0,
            tstop=90.0,
            rate_constant=params.get('rate', 10.0),
            location='proximal',
            weights_ampa={'L2_pyramidal': params.get('weight', 0.01)},
            weights_nmda={'L2_pyramidal': params.get('weight', 0.01)},
        )
    
    constraints = {
        'rate': (5.0, 20.0),
        'weight': (0.001, 0.05),
    }
    
    # Test with spectral power loss
    optimizer = Optimizer(
        initial_net=net,
        tstop=100.0,
        constraints=constraints,
        set_params=set_params_dummy,
        obj_fun=_spectral_power_loss,
        max_iter=5,  # Small number for testing
    )
    
    # Create mock target data
    target_freqs = np.linspace(1, 50, 50)
    target_psd = np.ones_like(target_freqs)
    
    # Should run without error
    optimizer.fit(
        target_psd=target_psd,
        target_freqs=target_freqs,
        freq_range=(5, 20),
        n_trials=1,
    )
    
    assert optimizer.net_ is not None
    assert len(optimizer.obj_) == 5
    assert optimizer.opt_params_ is not None


if __name__ == '__main__':
    test_custom_objective_function_validation()
    test_spectral_power_loss()
    test_phase_coherence_loss()
    test_custom_rmse_with_weights()
    test_optimizer_with_custom_loss()
    print("All tests passed!")