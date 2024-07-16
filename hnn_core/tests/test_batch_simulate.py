# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import pytest
import numpy as np
import os

from hnn_core.batch_simulate import BatchSimulate


@pytest.fixture
def batch_simulate_instance(tmp_path):
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

    return BatchSimulate(set_params=set_params,
                         tstop=1.,
                         file_path=tmp_path,
                         batch_size=3)


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
        param_grid)[:1]
    results = batch_simulate_instance.simulate_batch(param_combinations,
                                                     n_jobs=1,
                                                     backend='threading')
    assert len(results) == len(param_combinations)
    for result in results:
        assert 'net' in result
        assert 'dpl' in result
        assert 'param_values' in result

    # Validation Tests
    invalid_param_combinations = 'invalid'
    with pytest.raises(TypeError, match='param_combinations must be'):
        batch_simulate_instance.simulate_batch(invalid_param_combinations)

    with pytest.raises(TypeError, match='n_jobs must be'):
        batch_simulate_instance.simulate_batch(param_combinations,
                                               n_jobs='invalid')

    with pytest.raises(ValueError, match="Invalid value for the 'backend'"):
        batch_simulate_instance.simulate_batch(param_combinations,
                                               backend='invalid')

    with pytest.raises(TypeError, match='verbose must be'):
        batch_simulate_instance.simulate_batch(param_combinations,
                                               verbose='invalid')


def test_run(batch_simulate_instance, param_grid):
    """Test the run method of the batch_simulate_instance."""
    results_without_cache = batch_simulate_instance.run(param_grid,
                                                        n_jobs=2,
                                                        return_output=True,
                                                        combinations=False,
                                                        backend='loky',
                                                        clear_cache=False)

    total_combinations = len(
        batch_simulate_instance._generate_param_combinations(
            param_grid, combinations=False))

    assert results_without_cache is not None
    assert isinstance(results_without_cache, list)
    assert len(results_without_cache) == total_combinations

    results_with_cache = batch_simulate_instance.run(param_grid,
                                                     n_jobs=2,
                                                     return_output=True,
                                                     combinations=False,
                                                     backend='loky',
                                                     clear_cache=True)

    assert results_with_cache is not None
    assert isinstance(results_with_cache, list)
    assert len(results_with_cache) == 0

    # Validation Tests
    with pytest.raises(TypeError, match='param_grid must be'):
        batch_simulate_instance.run('invalid_param_grid')

    with pytest.raises(TypeError, match='n_jobs must be'):
        batch_simulate_instance.run(param_grid, n_jobs='invalid')

    with pytest.raises(ValueError, match="Invalid value for the 'backend'"):
        batch_simulate_instance.run(param_grid, backend='invalid_backend')

    with pytest.raises(TypeError, match='verbose must be'):
        batch_simulate_instance.run(param_grid, verbose='invalid')


def test_save_load_and_overwrite(batch_simulate_instance,
                                 param_grid, tmp_path):
    """Test the save method and its overwrite functionality."""
    param_combinations = batch_simulate_instance._generate_param_combinations(
        param_grid)[:3]
    results = batch_simulate_instance.simulate_batch(
        param_combinations,
        n_jobs=2)

    start_idx = 0
    end_idx = len(results)

    batch_simulate_instance._save(results, start_idx, end_idx)

    file_name = os.path.join(tmp_path, f'sim_run_{start_idx}-{end_idx}.npy')
    assert os.path.exists(file_name)

    loaded_results = np.load(file_name, allow_pickle=True)
    original_data = np.stack([dpl['dpl'][0].data['agg'] for dpl in results])
    assert (original_data == loaded_results).all()

    # Overwrite Test
    batch_simulate_instance.overwrite = False
    results[0]['dpl'][0].data['agg'][0] += 1

    with pytest.raises(FileExistsError):
        batch_simulate_instance._save(results, start_idx, end_idx)

    batch_simulate_instance.overwrite = True
    batch_simulate_instance._save(results, start_idx, end_idx)

    loaded_results = np.load(file_name, allow_pickle=True)

    original_data = np.stack([dpl['dpl'][0].data['agg'] for dpl in results])
    assert (original_data == loaded_results).all()

    # Validation Tests
    with pytest.raises(TypeError, match='results must be'):
        batch_simulate_instance._save('invalid_results', start_idx, end_idx)

    with pytest.raises(TypeError, match='start_idx must be'):
        batch_simulate_instance._save(results, 'invalid_start_idx', end_idx)

    with pytest.raises(TypeError, match='end_idx must be'):
        batch_simulate_instance._save(results, start_idx, 'invalid_end_idx')
