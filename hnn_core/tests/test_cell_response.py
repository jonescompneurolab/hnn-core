# Authors: Mainak Jas <mainakjas@gmail.com>

from glob import glob

import matplotlib.pyplot as plt
import pytest
import numpy as np

from hnn_core import CellResponse, read_spikes


def test_cell_response(tmp_path):
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
    net_attributes = ["_times", "_cell_type_names"]  # `Network.__init__`
    # creates these check that we always know which response attributes are
    # simulated see #291 for discussion; objective is to keep cell_response
    # size small
    assert sorted(list(cell_response.__dict__.keys())) == sorted(
        sim_attributes + net_attributes
    )

    # Test recovery of empty spike files
    empty_spike = CellResponse(
        cell_type_names=default_cell_type_names,
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
            spike_times=([2.3456, 7.89], [4.2812, 93.2]),
            spike_gids=spike_gids,
            spike_types=spike_types,
        )

    with pytest.raises(TypeError, match="spike_times should be a list of lists"):
        cell_response = CellResponse(
            cell_type_names=default_cell_type_names,
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
            spike_times=[[2.3456, 7.89]],
            spike_gids=spike_gids,
            spike_types=spike_types,
        )

    cell_response = CellResponse(
        cell_type_names=default_cell_type_names,
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

    # Write spike file with no 'types' column
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
    plt.close("all")
