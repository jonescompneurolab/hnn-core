# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import pytest

from hnn_core import Params
from hnn_core.feed import feed_event_times
from hnn_core.params import create_pext


def test_extfeed():
    """Test the different external feeds."""

    params = Params()
    p_common, p_unique = create_pext(params,
                                     params['tstop'])

    # feed name must be valid and unambiguous
    p_bogus = {'prng_seedcore': 0}
    pytest.raises(ValueError, feed_event_times,
                  'invalid_feed', None, p_bogus, 0)
    pytest.raises(ValueError, feed_event_times,
                  'ev', None, p_bogus, 0)  # ambiguous

    # 'unique' external feeds are always created
    for feed_type in ['extpois', 'extgauss']:
        event_times = feed_event_times(
            feed_type=feed_type,
            target_cell_type='L2_basket',
            params=p_unique[feed_type],
            gid=0)

    # but 'common' (rhythmic) feeds are not
    for ii in range(len(p_common)):  # len == 0 for def. params
        event_times = feed_event_times(
            feed_type='common',
            target_cell_type=None,
            params=p_common[ii],
            gid=0)
        # parameters should lead to 0 input spikes for default params
        assert len(event_times) == 0
        # check that ei.p_ext matches params
        loc = p_common[ii]['loc'][:4]  # loc=prox or dist
        for layer in ['L2', 'L5']:
            key = 'input_{}_A_weight_{}Pyr_ampa'.format(loc, layer)
            assert p_common[ii][layer + 'Pyr_ampa'][0] == params[key]

    # validate poisson input time interval
    params = p_unique['extpois']
    params['L2_basket'] = (1., 1., 0., 0.)
    with pytest.raises(ValueError, match='The end time for Poisson input'):
        params['t_interval'] = (params['t_interval'][0], -1)
        event_times = feed_event_times(
            feed_type='extpois',
            target_cell_type='L2_basket',
            params=params, gid=0)
    with pytest.raises(ValueError, match='The start time for Poisson'):
        params['t_interval'] = (-1, 5)
        event_times = feed_event_times(
            feed_type='extpois',
            target_cell_type='L2_basket',
            params=params, gid=0)

    # checks the distribution stats
    # if len(val_pois):
    #     xdiff = np.diff(val_pois/1000)
    #     print(lamtha, np.mean(xdiff), np.var(xdiff), 1/lamtha**2)
    # Convert array into nrn vector
    # if len(val_pois)>0: print('val_pois:',val_pois)
