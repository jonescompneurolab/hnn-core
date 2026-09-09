# Authors: Mainak Jas <mainakjas@gmail.com>

from glob import glob

import matplotlib.pyplot as plt
import pytest
import numpy as np

from hnn_core import CellResponse, read_spikes
from hnn_core.network_models import default_cell_metadata


@pytest.mark.parametrize(
    "input_metadata",
    [
        None,
        default_cell_metadata,
    ],
)
def test_cell_response(tmp_path, input_metadata):
    """Test CellResponse object."""
    # Round-trip test
    spike_times = [[2.3456, 7.89], [4.2812, 93.2]]
    spike_gids = [[1, 3], [5, 7]]
    spike_types = [["L2_pyramidal", "L2_basket"], ["L5_pyramidal", "L5_basket"]]
    tstart, tstop, fs = 0.1, 98.4, 1000.0
    sim_times = np.arange(tstart, tstop, 1 / fs)
    gid_ranges = {
        "L2_pyramidal": range(1, 2),
        "L2_basket": range(3, 4),
        "L5_pyramidal": range(5, 6),
        "L5_basket": range(7, 8),
    }
    default_cell_type_names = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]
    cell_response = CellResponse(
        cell_type_names=default_cell_type_names,
        cell_type_metadata=input_metadata,
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
        times=sim_times,
    )

    assert set(cell_response.cell_types) == set(gid_ranges.keys())
    assert cell_response.spike_times_by_type["L2_basket"] == [[7.89], []]
    assert cell_response.spike_times_by_type["L5_pyramidal"] == [[], [4.2812]]

    kwargs_hist = dict(alpha=0.25)
    fig = cell_response.plot_spikes_hist(show=False, **kwargs_hist)
    assert all(
        patch.get_alpha() == kwargs_hist["alpha"] for patch in fig.axes[0].patches
    ), "Alpha value not applied to all patches"

    # Testing writing using txt files
    cell_response.write(tmp_path / "spk_%d.txt")

    # Testing reading from txt files
    with pytest.warns(FutureWarning, match="Reading cell response"):
        assert cell_response == read_spikes(tmp_path / "spk_*.txt")

    assert "CellResponse | 2 simulation trials" in repr(cell_response)

    # reset clears all recorded variables, but leaves simulation time intact
    assert len(cell_response.times) == len(sim_times)
    sim_attributes = [
        "_spike_times",
        "_spike_gids",
        "_spike_types",
        "_vsec",
        "_isec",
        "_ca",
    ]
    net_attributes = [
        "_times",
        "_cell_type_names",
        "_cell_type_metadata",
    ]  # `Network.__init__`
    # creates these check that we always know which response attributes are
    # simulated see #291 for discussion; objective is to keep cell_response
    # size small
    assert sorted(list(cell_response.__dict__.keys())) == sorted(
        sim_attributes + net_attributes
    )

    # Test recovery of empty spike files
    empty_spike = CellResponse(
        cell_type_names=default_cell_type_names,
        cell_type_metadata=input_metadata,
        spike_times=[[], []],
        spike_gids=[[], []],
        spike_types=[[], []],
    )
    empty_spike.write(tmp_path / "empty_spk_%d.txt")
    empty_spike.write(tmp_path / "empty_spk.txt")
    empty_spike.write(tmp_path / "empty_spk_{0}.txt")
    assert empty_spike == read_spikes(tmp_path / "empty_spk_*.txt")

    assert "CellResponse | 2 simulation trials" in repr(empty_spike)

    with pytest.raises(TypeError, match="spike_times should be a list of lists"):
        cell_response = CellResponse(
            cell_type_names=default_cell_type_names,
            cell_type_metadata=input_metadata,
            spike_times=([2.3456, 7.89], [4.2812, 93.2]),
            spike_gids=spike_gids,
            spike_types=spike_types,
        )

    with pytest.raises(TypeError, match="spike_times should be a list of lists"):
        cell_response = CellResponse(
            cell_type_names=default_cell_type_names,
            cell_type_metadata=input_metadata,
            spike_times=[1, 2],
            spike_gids=spike_gids,
            spike_types=spike_types,
        )

    with pytest.raises(
        ValueError,
        match="spike times, gids, and types should be lists of the same length",
    ):
        cell_response = CellResponse(
            cell_type_names=default_cell_type_names,
            cell_type_metadata=input_metadata,
            spike_times=[[2.3456, 7.89]],
            spike_gids=spike_gids,
            spike_types=spike_types,
        )

    cell_response = CellResponse(
        cell_type_names=default_cell_type_names,
        cell_type_metadata=input_metadata,
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
    )

    with pytest.raises(
        TypeError, match="spike_types should be str, list, dict, or None"
    ):
        cell_response.plot_spikes_hist(spike_types=1, show=False)

    with pytest.raises(
        TypeError,
        match=r"spike_types\[ev\] must be a list\. "
        r"Got int\.",
    ):
        cell_response.plot_spikes_hist(spike_types={"ev": 1}, show=False)

    with pytest.raises(
        ValueError,
        match=r"Elements of spike_types must map to"
        r" mutually exclusive input types\. L2_basket is found"
        r" more than once\.",
    ):
        cell_response.plot_spikes_hist(
            spike_types={"ev": ["L2_basket", "L2_b"]}, show=False
        )

    with pytest.raises(ValueError, match="No input types found for ABC"):
        cell_response.plot_spikes_hist(spike_types="ABC", show=False)

    with pytest.raises(
        ValueError, match="tstart and tstop must be of type int or float"
    ):
        cell_response.mean_rates(tstart=0.1, tstop="ABC", gid_ranges=gid_ranges)

    with pytest.raises(ValueError, match="tstop must be greater than tstart"):
        cell_response.mean_rates(tstart=0.1, tstop=-1.0, gid_ranges=gid_ranges)

    with pytest.raises(
        ValueError,
        match="Invalid mean_type. Valid arguments include 'all', 'trial', or 'cell'.",
    ):
        cell_response.mean_rates(
            tstart=tstart, tstop=tstop, gid_ranges=gid_ranges, mean_type="ABC"
        )

    test_rate = (1 / (tstop - tstart)) * 1000

    assert cell_response.mean_rates(tstart, tstop, gid_ranges) == {
        "L5_pyramidal": test_rate / 2,
        "L5_basket": test_rate / 2,
        "L2_pyramidal": test_rate / 2,
        "L2_basket": test_rate / 2,
    }
    assert cell_response.mean_rates(tstart, tstop, gid_ranges, mean_type="trial") == {
        "L5_pyramidal": [0.0, test_rate],
        "L5_basket": [0.0, test_rate],
        "L2_pyramidal": [test_rate, 0.0],
        "L2_basket": [test_rate, 0.0],
    }
    assert cell_response.mean_rates(tstart, tstop, gid_ranges, mean_type="cell") == {
        "L5_pyramidal": [[0.0], [test_rate]],
        "L5_basket": [[0.0], [test_rate]],
        "L2_pyramidal": [[test_rate], [0.0]],
        "L2_basket": [[test_rate], [0.0]],
    }

    # repeat test for case in which one cell (L5 basket) does not spike -> L5_basket rate should be 0
    spike_times = [[2.3456, 7.89], [4.2812, 93.2]]
    spike_gids = [[1, 3], [5]]
    spike_types = [["L2_pyramidal", "L2_basket"], ["L5_pyramidal"]]
    tstart, tstop, fs = 0.1, 98.4, 1000.0

    test_rate = (1 / (tstop - tstart)) * 1000

    sim_times = np.arange(tstart, tstop, 1 / fs)
    gid_ranges = {
        "L2_pyramidal": range(1, 2),
        "L2_basket": range(3, 4),
        "L5_pyramidal": range(5, 6),
        "L5_basket": range(7, 8),
    }
    default_cell_type_names = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]
    cell_response = CellResponse(
        cell_type_names=default_cell_type_names,
        cell_type_metadata=input_metadata,
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
        times=sim_times,
    )

    assert cell_response.mean_rates(tstart=tstart, tstop=tstop) == {
        "L2_basket": test_rate / 2,
        "L2_pyramidal": test_rate / 2,
        "L5_basket": 0.0,
        "L5_pyramidal": test_rate / 2,
    }

    assert cell_response.mean_rates(tstart=tstart, tstop=tstop, mean_type="trial") == {
        "L2_basket": [test_rate, 0.0],
        "L2_pyramidal": [test_rate, 0.0],
        "L5_basket": [0.0, 0.0],
        "L5_pyramidal": [0.0, test_rate],
    }

    assert cell_response.mean_rates(tstart=tstart, tstop=tstop, mean_type="cell") == {
        "L2_basket": [[test_rate], [0.0]],
        "L2_pyramidal": [[test_rate], [0.0]],
        "L5_basket": [[0.0], [0.0]],
        "L5_pyramidal": [[0.0], [test_rate]],
    }

    # A cell_response with no spikes in any trial should yield 0 Hz rates for
    # every cell type when gid_ranges is unspecified (tests the empty
    # spike-record branch of CellResponse._gids_from_spikes)
    spike_times = [[], []]
    spike_gids = [[], []]
    spike_types = [[], []]
    cell_response = CellResponse(
        cell_type_names=default_cell_type_names,
        cell_type_metadata=input_metadata,
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
    )

    assert cell_response.mean_rates(tstart=tstart, tstop=tstop) == {
        "L2_basket": 0.0,
        "L2_pyramidal": 0.0,
        "L5_basket": 0.0,
        "L5_pyramidal": 0.0,
    }

    assert cell_response.mean_rates(tstart=tstart, tstop=tstop, mean_type="trial") == {
        "L2_basket": [0.0, 0.0],
        "L2_pyramidal": [0.0, 0.0],
        "L5_basket": [0.0, 0.0],
        "L5_pyramidal": [0.0, 0.0],
    }

    assert cell_response.mean_rates(tstart=tstart, tstop=tstop, mean_type="cell") == {
        "L2_basket": [[0.0], [0.0]],
        "L2_pyramidal": [[0.0], [0.0]],
        "L5_basket": [[0.0], [0.0]],
        "L5_pyramidal": [[0.0], [0.0]],
    }

    # Write spike file with no 'types' column
    spike_types = [["L2_pyramidal", "L2_basket"], ["L5_pyramidal", "L5_basket"]]
    for fname in sorted(glob(str(tmp_path / "spk_*.txt"))):
        times_gids_only = np.loadtxt(fname, dtype=str)[:, (0, 1)]
        np.savetxt(fname, times_gids_only, delimiter="\t", fmt="%s")

    # Check that spike_types are updated according to gid_ranges
    cell_response = read_spikes(tmp_path / "spk_*.txt", gid_ranges=gid_ranges)
    assert cell_response.spike_types == spike_types

    # Check for gid_ranges errors
    with pytest.raises(
        ValueError,
        match="gid_ranges must be provided if spike types are unspecified in the file ",
    ):
        cell_response = read_spikes(tmp_path / "spk_*.txt")
    with pytest.raises(
        ValueError, match="gid_ranges should contain only disjoint sets of gid values"
    ):
        gid_ranges = {
            "L2_pyramidal": range(3),
            "L2_basket": range(2, 4),
            "L5_pyramidal": range(4, 6),
            "L5_basket": range(6, 8),
        }
        cell_response = read_spikes(tmp_path / "spk_*.txt", gid_ranges=gid_ranges)

    # A gid that spikes more than once within a single trial should have all
    # of its spikes counted toward its rate (tests gid_counts > 1)
    spike_times = [[1.0, 2.0, 3.0]]
    spike_gids = [[1, 1, 3]]
    spike_types = [["L2_pyramidal", "L2_pyramidal", "L2_basket"]]
    tstart, tstop = 0.0, 10.0
    gid_ranges = {
        "L2_pyramidal": range(1, 2),
        "L2_basket": range(3, 4),
        "L5_pyramidal": range(5, 6),
        "L5_basket": range(7, 8),
    }
    cell_response = CellResponse(
        cell_type_names=default_cell_type_names,
        cell_type_metadata=input_metadata,
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
    )

    assert cell_response.mean_rates(tstart, tstop, gid_ranges) == {
        "L2_pyramidal": 200.0,
        "L2_basket": 100.0,
        "L5_pyramidal": 0.0,
        "L5_basket": 0.0,
    }

    assert cell_response.mean_rates(tstart, tstop, gid_ranges, mean_type="trial") == {
        "L2_pyramidal": [200.0],
        "L2_basket": [100.0],
        "L5_pyramidal": [0.0],
        "L5_basket": [0.0],
    }

    assert cell_response.mean_rates(tstart, tstop, gid_ranges, mean_type="cell") == {
        "L2_pyramidal": [[200.0]],
        "L2_basket": [[100.0]],
        "L5_pyramidal": [[0.0]],
        "L5_basket": [[0.0]],
    }

    plt.close("all")


def test_rate_over_time_trial_idx():
    """Test all non-validation code paths of rate_over_time.

    Covers trial_idx selection (int, list, out-of-order list, None), the
    cell_types forms (None, str, list, multiple types in one call), the
    zero-cells-recorded warning/all-zero branch, and normalization by
    n_cells when more than one gid of a type is present.
    """
    # 3 trials, single gid, with a distinct spike time per trial so each
    # trial's rate-over-time trace is distinguishable from the others
    spike_times = [[2.0], [8.0], [14.0]]
    spike_gids = [[7], [7], [7]]
    spike_types = [["L5_basket"], ["L5_basket"], ["L5_basket"]]
    tstart, tstop, fs = 0.0, 20.0, 10.0
    sim_times = np.arange(tstart, tstop, 1 / fs)

    cell_response = CellResponse(
        cell_type_names=["L5_basket"],
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
        times=sim_times,
    )

    window_length = 2.0

    # trial_idx=None: all trials computed at once
    rates_all = cell_response.rate_over_time(
        window_length=window_length, cell_types=["L5_basket"], trial_idx=None
    )
    assert rates_all["L5_basket"].shape == (3, len(sim_times))
    # sanity check that the 3 trials are not identical (i.e. the spikes at
    # different times actually produce different traces)
    assert not np.allclose(rates_all["L5_basket"][0], rates_all["L5_basket"][1])
    assert not np.allclose(rates_all["L5_basket"][1], rates_all["L5_basket"][2])

    # trial_idx as an integer must return that trial's own data
    for trial in (0, 1, 2):
        rate_int = cell_response.rate_over_time(
            window_length=window_length,
            cell_types=["L5_basket"],
            trial_idx=trial,
        )
        assert rate_int["L5_basket"].shape == (1, len(sim_times))
        assert np.allclose(rate_int["L5_basket"][0], rates_all["L5_basket"][trial])

    # trial_idx as a single-element list must return that trial's own data
    for trial in (0, 1, 2):
        rate_single = cell_response.rate_over_time(
            window_length=window_length,
            cell_types=["L5_basket"],
            trial_idx=[trial],
        )
        assert rate_single["L5_basket"].shape == (1, len(sim_times))
        assert np.allclose(rate_single["L5_basket"][0], rates_all["L5_basket"][trial])

    # trial_idx as a multi-element list, out of order, must preserve the
    # requested order in the output rows
    rate_multi = cell_response.rate_over_time(
        window_length=window_length, cell_types=["L5_basket"], trial_idx=[2, 0]
    )
    assert rate_multi["L5_basket"].shape == (2, len(sim_times))
    assert np.allclose(rate_multi["L5_basket"][0], rates_all["L5_basket"][2])
    assert np.allclose(rate_multi["L5_basket"][1], rates_all["L5_basket"][0])

    # trial_idx=None (default) must return an array of shape <n_trials, len(sim_times)>
    rate_all = cell_response.rate_over_time(
        window_length=window_length, cell_types=["L5_basket"]
    )
    n_trials = len(cell_response._spike_times)
    assert rate_all["L5_basket"].shape == (n_trials, len(sim_times))

    # raise error if cell_types input contains type not in cell_response
    # cell_types=None must default to every name in cell_type_names
    with pytest.raises(ValueError, match="Invalid cell type provided."):
        _ = cell_response.rate_over_time(
            window_length=window_length, cell_types=["L5_basket", "L5_pyramidal"]
        )

    # add cell type that doesn't spike
    cell_response_multi = CellResponse(
        cell_type_names=["L5_basket", "L5_pyramidal"],
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
        times=sim_times,
    )

    rates_multi = cell_response_multi.rate_over_time(window_length=window_length)
    assert set(rates_multi.keys()) == {"L5_basket", "L5_pyramidal"}
    # ensure L5_basket spikes are still calculated correctly
    assert np.allclose(rates_multi["L5_basket"], rates_all["L5_basket"])
    # ensure L5_pyramidal is all 0
    assert np.array_equal(rates_multi["L5_pyramidal"], np.zeros((3, len(sim_times))))

    # cell_types as a bare str must behave like a single-element list
    rate_str = cell_response_multi.rate_over_time(
        window_length=window_length, cell_types="L5_basket"
    )
    assert set(rate_str.keys()) == {"L5_basket"}
    assert np.allclose(rate_str["L5_basket"], rates_all["L5_basket"])

    # cell_types as a list with more than one entry, requested together,
    # must return the same per-type results as requesting them separately
    rate_both = cell_response_multi.rate_over_time(
        window_length=window_length, cell_types=["L5_basket", "L5_pyramidal"]
    )
    assert set(rate_both.keys()) == {"L5_basket", "L5_pyramidal"}
    assert np.allclose(rate_both["L5_basket"], rates_all["L5_basket"])
    assert np.array_equal(rate_both["L5_pyramidal"], np.zeros((3, len(sim_times))))


def test_rate_over_time_validation():
    """Test input validation of rate_over_time, independent of its outputs."""
    spike_times = [[2.0], [8.0]]
    spike_gids = [[7], [7]]
    spike_types = [["L5_basket"], ["L5_basket"]]
    tstart, tstop, fs = 0.0, 20.0, 10.0
    sim_times = np.arange(tstart, tstop, 1 / fs)
    total_duration = sim_times[-1] - sim_times[0]
    dt = np.diff(sim_times)[0]

    cell_response = CellResponse(
        cell_type_names=["L5_basket"],
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
        times=sim_times,
    )

    # times must have at least two entries with non-zero spacing
    bad_times_response = CellResponse(
        cell_type_names=["L5_basket"],
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
        times=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="'times' must contain at least two"):
        bad_times_response.rate_over_time(window_length=2.0)

    # window_length must be int or float
    with pytest.raises(TypeError, match="window_length"):
        cell_response.rate_over_time(window_length="2.0")

    # window_length must be greater than dt
    with pytest.raises(ValueError, match="greater than the 'dt'"):
        cell_response.rate_over_time(window_length=dt)

    # window_length must not exceed the total recorded duration
    with pytest.raises(ValueError, match="not be greater than the total duration"):
        cell_response.rate_over_time(window_length=total_duration + 1.0)

    # trial_idx must be int, list, or None
    with pytest.raises(TypeError, match="trial_idx must be an instance of"):
        cell_response.rate_over_time(window_length=2.0, trial_idx="goof")

    # trial_idx as an out-of-range int
    with pytest.raises(ValueError, match="'trial_idx' must be a non-negative"):
        cell_response.rate_over_time(window_length=2.0, trial_idx=5)
    with pytest.raises(ValueError, match="'trial_idx' must be a non-negative"):
        cell_response.rate_over_time(window_length=2.0, trial_idx=-1)

    # trial_idx list with non-integer elements
    with pytest.raises(ValueError, match="must be integers"):
        cell_response.rate_over_time(window_length=2.0, trial_idx=[0, "1"])

    # trial_idx list with out-of-range elements
    with pytest.raises(ValueError, match="less than the number of trials"):
        cell_response.rate_over_time(window_length=2.0, trial_idx=[0, 5])

    # trial_idx as an empty list
    with pytest.raises(ValueError, match="'trial_idx' must not be an empty list"):
        cell_response.rate_over_time(window_length=2.0, trial_idx=[])

    # cell_types must be str, list, or None
    with pytest.raises(TypeError, match="cell_types"):
        cell_response.rate_over_time(window_length=2.0, cell_types=5)

    # cell_types as an invalid string
    with pytest.raises(ValueError, match="Invalid cell type provided"):
        cell_response.rate_over_time(window_length=2.0, cell_types="not_a_cell_type")

    # cell_types list containing an invalid entry
    with pytest.raises(ValueError, match="Invalid cell type provided"):
        cell_response.rate_over_time(
            window_length=2.0, cell_types=["L5_basket", "not_a_cell_type"]
        )

    # cell_types as an empty list
    with pytest.raises(ValueError, match="'cell_types' must not be an empty list"):
        cell_response.rate_over_time(window_length=2.0, cell_types=[])
