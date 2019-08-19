"""Reproduce classic Okada constant paper.

References
----------
[1] Invariance in current dipole moment density across brain structures and
species: Physiological constraint for neuroimaging

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

import numpy as np
import matplotlib.pyplot as plt

from mne_neuron.pyramidal import L5Pyr
from mne_neuron.basket import Basket

####################################################################
# Let us first define a list of 100 Pyramidal and 100 Basket neurons

pyrs, baskets = list(), list()
for gid in range(100):
    pyrs.append(L5Pyr(gid=gid))
    baskets.append(Basket(gid=gid + 100, pos=0))

####################################################################
# Then we construct a connectivity matrix
co_pyr_basket = np.zeros((100, 100))
co_basket_pyr = np.zeros((100, 100))
for idx in range(100):
    target_idx = np.random.choice(range(100), size=20)
    co_pyr_basket[idx, target_idx] = 1.
    target_idx = np.random.choice(range(100), size=20)
    co_basket_pyr[idx, target_idx] = 1.
plt.matshow(co_pyr_basket)
plt.matshow(co_basket_pyr)
plt.show()

####################################################################
# Now we connect them

postsyn = pyrs[0].synapses['soma_gabaa']
nc_dict = {
    'A_delay': 1.,
    'lamtha': 3.0,
    'threshold': 0.0,
    'type_src': 'L5_Pyramidal'
}

pyrs[0].parconnect_from_src(gid_presyn=100, nc_dict=nc_dict,
                            postsyn=pyrs[0].synapses['soma_gabaa'])
