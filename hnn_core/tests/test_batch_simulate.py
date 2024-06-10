# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import pytest
import numpy as np

from hnn_core import BatchSimulate


@pytest.fixture
def batch_simulate_instance():
    """Fixture for creating a BatchSimulate instance with custom parameters."""
    def set_params(param_values, net):
        weights_ampa = {'L2_basket': param_values['weight_basket'],
                        'L2_pyramidal': param_values['weight_pyr'],
                        'L5_basket': param_values['weight_basket'],
                        'L5_pyramidal': param_values['weight_pyr']}

        synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                           'L5_basket': 1., 'L5_pyramidal': 1.}

        mu = param_values['mu']
        sigma = param_values['sigma']
        net.add_evoked_drive('evprox',
                             mu=mu,
                             sigma=sigma,
                             numspikes=1,
                             location='proximal',
                             weights_ampa=weights_ampa,
                             synaptic_delays=synaptic_delays)

    return BatchSimulate(set_params=set_params, tstop=10.)


@pytest.fixture
def param_grid():
    """Returns a dictionary representing a parameter grid for
    batch simulation."""
    return {
        'weight_basket': np.logspace(-4 - 1, 2),
        'weight_pyr': np.logspace(-4, -1, 2),
        'mu': np.linspace(20, 80, 2),
        'sigma': np.linspace(1, 20, 2)
    }


def test_generate_param_combinations(batch_simulate_instance, param_grid):
    """Test generating parameter combinations."""
    param_combinations = batch_simulate_instance._generate_param_combinations(
        param_grid)
    assert len(param_combinations) == (
        len(param_grid['weight_basket']) *
        len(param_grid['weight_pyr']) *
        len(param_grid['mu']) *
        len(param_grid['sigma'])
    )


def test_run_single_sim(batch_simulate_instance):
    """Test running a single simulation."""
    param_values = {
        'weight_basket': -3,
        'weight_pyr': -2,
        'mu': 40,
        'sigma': 20
    }
    result = batch_simulate_instance._run_single_sim(param_values)
    assert 'net' in result
    assert 'dpl' in result
    assert 'param_values' in result
    assert result['param_values'] == param_values


def test_simulate_batch(batch_simulate_instance, param_grid):
    """Test simulating a batch of parameter sets."""
    param_combinations = batch_simulate_instance._generate_param_combinations(
        param_grid)[:3]
    results = batch_simulate_instance.simulate_batch(param_combinations,
                                                     n_jobs=2)
    assert len(results) == len(param_combinations)
    for result in results:
        assert 'net' in result
        assert 'dpl' in result
        assert 'param_values' in result


def test_run(batch_simulate_instance, param_grid):
    """Test the run method of the batch_simulate_instance."""
    results = batch_simulate_instance.run(param_grid, n_jobs=2,
                                          return_output=True,
                                          combinations=False)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == len(
        batch_simulate_instance._generate_param_combinations(
            param_grid, combinations=False))
