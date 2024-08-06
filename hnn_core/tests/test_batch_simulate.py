# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import pytest
import numpy as np
import os

from hnn_core.batch_simulate import BatchSimulate
from hnn_core import jones_2009_model


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

    net = jones_2009_model()
    return BatchSimulate(net=net, set_params=set_params,
                         tstop=1.,
                         save_folder=tmp_path,
                         batch_size=3)


@pytest.fixture
def param_grid():
    """Returns a dictionary representing a parameter grid for
    batch simulation."""
    return {
        'weight_basket': np.logspace(-4, -1, 2),
        'weight_pyr': np.logspace(-4, -1, 2),
        'mu': [40],
        'sigma': [5]
    }


def test_parameter_validation():
    boolean_params = [
        'save_outputs',
        'save_dpl',
        'save_spiking',
        'save_lfp',
        'save_voltages',
        'save_currents',
        'save_calcium',
        'clear_cache',
        'summary_func'
    ]

    for param in boolean_params:
        with pytest.raises(TypeError, match=f'{param} must be'):
            BatchSimulate(set_params=lambda x: x, **{param: 'invalid'})

    with pytest.raises(TypeError, match='set_params must be'):
        BatchSimulate(set_params='invalid')

    with pytest.raises(TypeError, match="net must be"):
        BatchSimulate(net="invalid_network", set_params=lambda x: x)


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
    assert isinstance(result['net'], type(batch_simulate_instance.net))


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
                                                        backend='loky')

    total_combinations = len(
        batch_simulate_instance._generate_param_combinations(
            param_grid, combinations=False))

    assert results_without_cache is not None
    assert isinstance(results_without_cache, dict)
    assert 'simulated_data' in results_without_cache
    assert len(results_without_cache['simulated_data']
               ) == total_combinations

    batch_simulate_instance.clear_cache = True
    results_with_cache = batch_simulate_instance.run(param_grid,
                                                     n_jobs=2,
                                                     return_output=True,
                                                     combinations=False,
                                                     backend='loky',
                                                     verbose=50)

    assert results_with_cache is not None
    assert isinstance(results_with_cache, dict)
    assert 'summary_statistics' in results_with_cache

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

    file_name = os.path.join(tmp_path, f'sim_run_{start_idx}-{end_idx}.npz')
    assert os.path.exists(file_name)

    loaded_data = np.load(file_name, allow_pickle=True)
    loaded_results = {key: loaded_data[key].tolist()
                      for key in loaded_data.files}

    original_data = np.stack([result['dpl'][0].data['agg']
                              for result in results])
    loaded_data = np.stack([dpl[0].data['agg']
                            for dpl in loaded_results['dpl']])

    assert (original_data == loaded_data).all()

    # Overwrite Test
    batch_simulate_instance.overwrite = False
    results[0]['dpl'][0].data['agg'][0] += 1

    with pytest.raises(FileExistsError):
        batch_simulate_instance._save(results, start_idx, end_idx)

    batch_simulate_instance.overwrite = True
    batch_simulate_instance._save(results, start_idx, end_idx)

    loaded_data = np.load(file_name, allow_pickle=True)
    loaded_results = {key: loaded_data[key].tolist()
                      for key in loaded_data.files}

    original_data = np.stack([result['dpl'][0].data['agg']
                              for result in results])
    loaded_data = np.stack([dpl[0].data['agg']
                            for dpl in loaded_results['dpl']])

    assert (original_data == loaded_data).all()

    # Validation Tests
    with pytest.raises(TypeError, match='results must be'):
        batch_simulate_instance._save('invalid_results', start_idx, end_idx)

    with pytest.raises(TypeError, match='start_idx must be'):
        batch_simulate_instance._save(results, 'invalid_start_idx', end_idx)

    with pytest.raises(TypeError, match='end_idx must be'):
        batch_simulate_instance._save(results, start_idx, 'invalid_end_idx')


def test_load_results(batch_simulate_instance, param_grid, tmp_path):
    """Test loading results from a single file."""
    param_combinations = batch_simulate_instance._generate_param_combinations(
        param_grid)[:3]
    results = batch_simulate_instance.simulate_batch(
        param_combinations,
        n_jobs=2)

    start_idx = 0
    end_idx = len(results)
    batch_simulate_instance._save(results, start_idx, end_idx)

    file_name = os.path.join(tmp_path, f'sim_run_{start_idx}-{end_idx}.npz')
    assert os.path.exists(file_name)

    # single result file
    loaded_results = batch_simulate_instance.load_results(file_name)
    assert 'param_values' in loaded_results
    assert 'dpl' in loaded_results
    assert len(loaded_results['dpl']) == len(results)

    original_data = np.stack([result['dpl'][0].data['agg']
                              for result in results])
    loaded_data = np.stack([dpl[0].data['agg']
                            for dpl in loaded_results['dpl']])
    assert np.array_equal(original_data, loaded_data)

    for key in ['spiking', 'lfp', 'voltages', 'currents', 'calcium']:
        assert key not in loaded_results

    # all result files
    all_loaded_results = batch_simulate_instance.load_all_results()
    assert len(all_loaded_results) == 1

    all_loaded_data = np.stack([dpl[0].data['agg']
                               for dpl in all_loaded_results[0]['dpl']])
    assert np.array_equal(original_data, all_loaded_data)

    # Validation Tests
    with pytest.raises(TypeError, match='results must be'):
        batch_simulate_instance._save("invalid_results", start_idx, end_idx)
