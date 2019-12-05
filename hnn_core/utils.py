"""Utils."""

# Authors: Mainak Jas <mjas@harvard.mgh.edu>

import platform
import os.path as op

from neuron import h


def load_custom_mechanisms():
    if platform.system() == 'Windows':
        mech_fname = op.join(op.dirname(__file__), '..', 'mod', 'nrnmech.dll')
    else:
        mech_fname = op.join(op.dirname(__file__), '..', 'mod', 'x86_64',
                             '.libs', 'libnrnmech.so')
    h.nrn_load_dll(mech_fname)
    print('Loading custom mechanism files from %s' % mech_fname)
