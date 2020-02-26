import os.path as op

from numpy import loadtxt
from numpy.testing import assert_array_equal

from mne.utils import _fetch_file
import hnn_core
from hnn_core import simulate_dipole, Network, read_params, get_rank


def test_hnn_core(n_jobs=1):
    """Test to check if hnn-core does not break."""
    if get_rank() == 0:
        # small snippet of data on data branch for now. To be deleted
        # later. Data branch should have only commit so it does not
        # pollute the history.
        data_url = ('https://raw.githubusercontent.com/jonescompneurolab/'
                    'hnn-core/test_data/dpl.txt')
        if not op.exists('dpl.txt'):
            _fetch_file(data_url, 'dpl.txt')
        dpl_master = loadtxt('dpl.txt')

    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

    # default params
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    # run the simulation
    net = Network(params)
    dpl = simulate_dipole(net, n_jobs=n_jobs)[0]

    if get_rank() == 0:
        # write the dipole to a file and compare
        fname = './dpl2.txt'
        dpl.write(fname)

        dpl_pr = loadtxt(fname)
        assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
        assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5

        # Test spike type counts
        spiketype_counts = {}
        for spikegid in net.spikes.gids[0]:
            if net.gid_to_type(spikegid) not in spiketype_counts:
                spiketype_counts[net.gid_to_type(spikegid)] = 0
            else:
                spiketype_counts[net.gid_to_type(spikegid)] += 1
        assert 'common' not in spiketype_counts
        assert 'exgauss' not in spiketype_counts
        assert 'extpois' not in spiketype_counts
        assert spiketype_counts == {'evprox1': 269,
                                    'L2_basket': 54,
                                    'L2_pyramidal': 113,
                                    'L5_pyramidal': 395,
                                    'L5_basket': 85,
                                    'evdist1': 234,
                                    'evprox2': 269}
