"""Handling of parameters."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import json
import fnmatch
import os.path as op
from copy import deepcopy


def _read_json(fname):
    """Read param values from a .json file.
    Parameters
    ----------
    fname : str
        Full path to the file (.json)

    Returns
    -------
    params_input : dict
        Dictionary of parameters
    """
    with open(fname) as json_data:
        params_input = json.load(json_data)

    return params_input


def read_params(params_fname):
    """Read param values from a file (.json or .param).

    Parameters
    ----------
    params_fname : str
        Full path to the file (.param)

    Returns
    -------
    params : an instance of Params
        Params containing parameter values from file
    """

    split_fname = op.splitext(params_fname)
    ext = split_fname[1]

    if ext == '.json':
        params_dict = _read_json(params_fname)
    else:
        raise ValueError('Unrecognized extension, expected one of' +
                         ' .json, .param. Got %s' % ext)

    if len(params_dict) == 0:
        raise ValueError("Failed to read parameters from file: %s" %
                         op.normpath(params_fname))

    params = Params(params_dict)

    return params


class Params(dict):
    """Params object.
    Parameters
    ----------
    params_input : dict
        Dictionary of parameters.
    """

    def __repr__(self):
        """Display the params nicely."""
        return json.dumps(self, sort_keys=True, indent=4)

    def __getitem__(self, key):
        """Return a subset of parameters."""
        keys = self.keys()
        if key in keys:
            return dict.__getitem__(self, key)
        else:
            matches = fnmatch.filter(keys, key)
            if len(matches) == 0:
                return dict.__getitem__(self, key)
            params = self.copy()
            for key in keys:
                if key not in matches:
                    params.pop(key)
            return params

    def __setitem__(self, key, value):
        """Set the value for a subset of parameters."""
        keys = self.keys()
        if key in keys:
            return dict.__setitem__(self, key, value)
        else:
            matches = fnmatch.filter(keys, key)
            if len(matches) == 0:
                return dict.__setitem__(self, key, value)
            for key in keys:
                if key in matches:
                    self.update({key: value})

    def copy(self):
        return deepcopy(self)

    def write(self, fname):

        """Write param values to a file.
        Parameters
        ----------
        fname : str
            Full path to the output file (.json)
        """
        with open(fname, 'w') as fp:
            json.dump(self, fp)

# debug test function
if __name__ == '__main__':
    fparam = 'param/debug.param'
