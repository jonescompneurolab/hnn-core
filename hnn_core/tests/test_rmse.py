import os.path as op

from numpy import loadtxt, c_

from mne.utils import _fetch_file
import hnn_core
from hnn_core import simulate_dipole, average_dipoles, rmse
from hnn_core import Params, Network, Dipole


def test_rmse():
    """Test to check RMSE calculation"""
    data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/'
                'master/data/MEG_detection_data/yes_trial_S1_ERP_all_avg.txt')
    if not op.exists('yes_trial_S1_ERP_all_avg.txt'):
        _fetch_file(data_url, 'yes_trial_S1_ERP_all_avg.txt')
    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

    exp_dpl = Dipole(extdata[:, 0], c_[extdata[:, 1]])

    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = Params(params_fname)

    net = Network(params)
    dpls = simulate_dipole(net)
    avg_dpl = average_dipoles(dpls)
    avg_rmse = rmse(avg_dpl, exp_dpl, tstop=params['tstop'])
    expected_rmse = 4.533252902006792

    assert(round(avg_rmse, 12) == round(expected_rmse, 12))
