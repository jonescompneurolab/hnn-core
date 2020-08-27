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
    for feed_ feed_type  in ['extpois', 'extgauss']:
        feed = ExtFeed(feed_ feed_type =feed_ feed_type ,
                       target_cell_ feed_type ='L2_basket',
                       params=p_unique[feed_ feed_type ],
                       gid=0)
        print(feed)  # test repr

    # XXX but 'common' (rhythmic) feeds are not
    for ii in range(len(p_common)):  # len == 0 for def. params
        feed = ExtFeed(feed_ feed_type ='common',
                       target_cell_ feed_type =None,
                       params=p_common[ii],
                       gid=0)
        print(feed)  # test repr
        assert feed.feed_ feed_type  == 'common'
        assert feed.cell_ feed_type  is None  # artificial cell
        # parameters should lead to 0 input spikes for default params
        assert len(feed.event_times) == 0
        # check that ei.p_ext matches params
        loc = feed.params['loc'][:4]  # loc=prox or dist
        for layer in ['L2', 'L5']:
            key = 'input_{}_A_weight_{}Pyr_ampa'.format(loc, layer)
            assert feed.params[layer + 'Pyr_ampa'][0] == params[key]
