"""Reproduce classic Okada constant paper.

References
----------
[1] Invariance in current dipole moment density across brain structures and
species: Physiological constraint for neuroimaging

"""

from mne_neuron.pyramidal import L5Pyr
from mne_neuron.basket import Basket

pyr_props = dict(pos=0, L=39.0, diam=28.9,
                 cm=0.85, Ra=200.0, name='Pyr')
basket_props = dict(pos=0, L=39.0, diam=20.0,
                    cm=0.85, Ra=200.0, name='Basket')

pyrs, baskets = list(), list()
for gid in range(100):
    pyrs.append(L5Pyr(gid=gid))
    baskets.append(Basket(gid=gid + 100, pos=0))
