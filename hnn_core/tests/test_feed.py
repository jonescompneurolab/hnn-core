# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import pytest

from hnn_core import Params
from hnn_core.feed import ExtFeed
from hnn_core.params import create_pext


def test_extfeed():
    """Test the different external feeds."""

    params = Params()
    p_common, p_unique = create_pext(params,
                                     params['tstop'])

    # feed name must be valid and unambiguous
    p_bogus = {'prng_seedcore': 0}
    pytest.raises(ValueError, ExtFeed,
                  'invalid_feed', None, p_bogus, 0)
    pytest.raises(ValueError, ExtFeed,
                  'ev', None, p_bogus, 0)  # ambiguous

    # XXX 'unique' external feeds are always created; why?
    for feed_type in ['extpois', 'extgauss']:
        feed = ExtFeed(feed_type=feed_type,
                       target_cell_type='L2_basket',
                       params=p_unique[feed_type],
                       gid=0)
        print(feed)  # test repr

    # XXX but 'common' (rhythmic) feeds are not
    for ii in range(len(p_common)):  # len == 0 for def. params
        feed = ExtFeed(feed_type='common',
                       target_cell_type=None,
                       params=p_common[ii],
                       gid=0)
        print(feed)  # test repr
        assert feed.feed_type == 'common'
        assert feed.cell_type is None  # artificial cell
        # parameters should lead to 0 input spikes for default params
        assert len(feed.event_times) == 0
        # check that ei.p_ext matches params
        loc = feed.params['loc'][:4]  # loc=prox or dist
        for layer in ['L2', 'L5']:
            key = 'input_{}_A_weight_{}Pyr_ampa'.format(loc, layer)
            assert feed.params[layer + 'Pyr_ampa'][0] == params[key]

    # validate poisson input time interval
    params = p_unique['extpois']
    params['L2_basket'] = (1., 1., 0., 0.)
    with pytest.raises(ValueError, match='The end time for Poisson input'):
        params['t_interval'] = (params['t_interval'][0], -1)
        feed = ExtFeed(feed_type='extpois',
                       target_cell_type='L2_basket',
                       params=params, gid=0)
    with pytest.raises(ValueError, match='The start time for Poisson'):
        params['t_interval'] = (-1, 5)
        feed = ExtFeed(feed_type='extpois',
                       target_cell_type='L2_basket',
                       params=params, gid=0)
