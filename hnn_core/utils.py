"""Utils."""

# Authors: Mainak Jas <mjas@harvard.mgh.edu>

import platform
import os.path as op

from neuron import h

# NEURON only allows mechanisms to be loaded once (per Python interpreter)
_loaded_dll = None


def load_custom_mechanisms():
    global _loaded_dll

    if _loaded_dll is not None:
        return

    from .parallel import get_rank

    if platform.system() == 'Windows':
        mech_fname = op.join(op.dirname(__file__), '..', 'mod', 'nrnmech.dll')
    else:
        mech_fname = op.join(op.dirname(__file__), '..', 'mod', 'x86_64',
                             '.libs', 'libnrnmech.so')
    h.nrn_load_dll(mech_fname)
    _loaded_dll = mech_fname

    if get_rank() == 0:
        print('Loading custom mechanism files from %s' % mech_fname)

    return
