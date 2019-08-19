# Authors: Mainak Jas <mainakjas@gmail.com>

from mne_neuron.pyramidal import L5Pyr
from mne_neuron.basket import Basket


def test_cell():
    """Test cell object."""
    pyr = L5Pyr(gid=0)
    assert 'gid' in repr(pyr)

    basket = Basket(gid=100)
    assert 'gid' in repr(basket)
