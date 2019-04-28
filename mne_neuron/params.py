"""Handling of parameters."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import json
import fnmatch
from copy import deepcopy

from .params_default import get_params_default


# return number of evoked inputs (proximal, distal)
# using dictionary d (or if d is a string, first load the dictionary from
# filename d)
def _count_evoked_inputs(d):
    nprox = ndist = 0
    for k, v in d.items():
        if k.startswith('t_'):
            if k.count('evprox') > 0:
                nprox += 1
            elif k.count('evdist') > 0:
                ndist += 1
    return nprox, ndist


class Params(dict):
    """Params object.

    Parameters
    ----------
    params_fname : str
        The parameters json file
    """

    def __init__(self, params_fname):
        with open(params_fname) as json_data:
            params_input = json.load(json_data)

        nprox, ndist = _count_evoked_inputs(params_input)

        # create a copy of params_default through which to iterate
        params = get_params_default(nprox, ndist)

        for key in params.keys():
            if key in params_input:
                params[key] = params_input.pop(key)
            self[key] = params[key]

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

def feed_validate(p_ext, d, tstop):
    """Whips into shape ones that are not
       could be properly made into a meaningful class.
    """
    # only append if t0 is less than simulation tstop
    if tstop > d['t0']:
        # # reset tstop if the specified tstop exceeds the
        # # simulation runtime
        # if d['tstop'] == 0:
        #     d['tstop'] = tstop

        if d['tstop'] > tstop:
            d['tstop'] = tstop

        # if stdev is zero, increase synaptic weights 5 fold to make
        # single input equivalent to 5 simultaneous input to prevent spiking
        # <<---- SN: WHAT IS THIS RULE!?!?!?
        if not d['stdev'] and d['distribution'] != 'uniform':
            for key in d.keys():
                if key.endswith('Pyr'):
                    d[key] = (d[key][0] * 5., d[key][1])
                elif key.endswith('Basket'):
                    d[key] = (d[key][0] * 5., d[key][1])

        # if L5 delay is -1, use same delays as L2 unless L2 delay is 0.1 in
        # which case use 1. <<---- SN: WHAT IS THIS RULE!?!?!?
        if d['L5Pyr_ampa'][1] == -1:
            for key in d.keys():
                if key.startswith('L5'):
                    if d['L2Pyr'][1] != 0.1:
                        d[key] = (d[key][0], d['L2Pyr'][1])
                    else:
                        d[key] = (d[key][0], 1.)

        p_ext.append(d)

    return p_ext


def check_evoked_synkeys(p, nprox, ndist):
    # make sure ampa,nmda gbar values are in the param dict for evoked
    # inputs(for backwards compatibility)
    # evoked distal target cell types
    lctprox = ['L2Pyr', 'L5Pyr', 'L2Basket', 'L5Basket']
    # evoked proximal target cell types
    lctdist = ['L2Pyr', 'L5Pyr', 'L2Basket']
    lsy = ['ampa', 'nmda']  # synapse types used in evoked inputs
    for nev, pref, lct in zip([nprox, ndist], ['evprox_', 'evdist_'],
                              [lctprox, lctdist]):
        for i in range(nev):
            skey = pref + str(i + 1)
            for sy in lsy:
                for ct in lct:
                    k = 'gbar_' + skey + '_' + ct + '_' + sy
                    # if the synapse-specific gbar not present, use the
                    # existing weight for both ampa,nmda
                    if k not in p:
                        p[k] = p['gbar_' + skey + '_' + ct]

#


def check_pois_synkeys(p):
    # make sure ampa,nmda gbar values are in the param dict for Poisson inputs
    # (for backwards compatibility)
    lct = ['L2Pyr', 'L5Pyr', 'L2Basket', 'L5Basket']  # target cell types
    lsy = ['ampa', 'nmda']  # synapse types used in Poisson inputs
    for ct in lct:
        for sy in lsy:
            k = ct + '_Pois_A_weight_' + sy
            # if the synapse-specific weight not present, set it to 0 in p
            if k not in p:
                p[k] = 0.0

# creates the external feed params based on individual simulation params p


def create_pext(p, tstop):
    """Indexable Python list of param dicts for parallel.

    Turn off individual feeds by commenting out relevant line here.
    always valid, no matter the length.

    Parameters
    ----------
    p : dict
        The parameters returned by ExpParams(f_psim).return_pdict()
    """
    p_ext = []

    # p_unique is a dict of input param types that end up going to each cell
    # uniquely
    p_unique = {}

    # default params for proximal rhythmic inputs
    feed_prox = {
        'f_input': p['f_input_prox'],
        't0': p['t0_input_prox'],
        'tstop': p['tstop_input_prox'],
        'stdev': p['f_stdev_prox'],
        'L2Pyr_ampa': (p['input_prox_A_weight_L2Pyr_ampa'],
                       p['input_prox_A_delay_L2']),
        'L2Pyr_nmda': (p['input_prox_A_weight_L2Pyr_nmda'],
                       p['input_prox_A_delay_L2']),
        'L5Pyr_ampa': (p['input_prox_A_weight_L5Pyr_ampa'],
                       p['input_prox_A_delay_L5']),
        'L5Pyr_nmda': (p['input_prox_A_weight_L5Pyr_nmda'],
                       p['input_prox_A_delay_L5']),
        'L2Basket_ampa': (p['input_prox_A_weight_L2Basket_ampa'],
                          p['input_prox_A_delay_L2']),
        'L2Basket_nmda': (p['input_prox_A_weight_L2Basket_nmda'],
                          p['input_prox_A_delay_L2']),
        'L5Basket_ampa': (p['input_prox_A_weight_L5Basket_ampa'],
                          p['input_prox_A_delay_L5']),
        'L5Basket_nmda': (p['input_prox_A_weight_L5Basket_nmda'],
                          p['input_prox_A_delay_L5']),
        'events_per_cycle': p['events_per_cycle_prox'],
        'prng_seedcore': int(p['prng_seedcore_input_prox']),
        'distribution': p['distribution_prox'],
        'lamtha': 100.,
        'loc': 'proximal',
        'repeats': p['repeats_prox'],
        't0_stdev': p['t0_input_stdev_prox'],
        'threshold': p['threshold']
    }

    # ensures time interval makes sense
    p_ext = feed_validate(p_ext, feed_prox, tstop)

    # default params for distal rhythmic inputs
    feed_dist = {
        'f_input': p['f_input_dist'],
        't0': p['t0_input_dist'],
        'tstop': p['tstop_input_dist'],
        'stdev': p['f_stdev_dist'],
        'L2Pyr_ampa': (p['input_dist_A_weight_L2Pyr_ampa'],
                       p['input_dist_A_delay_L2']),
        'L2Pyr_nmda': (p['input_dist_A_weight_L2Pyr_nmda'],
                       p['input_dist_A_delay_L2']),
        'L5Pyr_ampa': (p['input_dist_A_weight_L5Pyr_ampa'],
                       p['input_dist_A_delay_L5']),
        'L5Pyr_nmda': (p['input_dist_A_weight_L5Pyr_nmda'],
                       p['input_dist_A_delay_L5']),
        'L2Basket_ampa': (p['input_dist_A_weight_L2Basket_ampa'],
                          p['input_dist_A_delay_L2']),
        'L2Basket_nmda': (p['input_dist_A_weight_L2Basket_nmda'],
                          p['input_dist_A_delay_L2']),
        'events_per_cycle': p['events_per_cycle_dist'],
        'prng_seedcore': int(p['prng_seedcore_input_dist']),
        'distribution': p['distribution_dist'],
        'lamtha': 100.,
        'loc': 'distal',
        'repeats': p['repeats_dist'],
        't0_stdev': p['t0_input_stdev_dist'],
        'threshold': p['threshold']
    }

    p_ext = feed_validate(p_ext, feed_dist, tstop)

    nprox, ndist = _count_evoked_inputs(p)
    # print('nprox,ndist evoked inputs:', nprox, ndist)

    # NEW: make sure all evoked synaptic weights present
    # (for backwards compatibility)
    # could cause differences between output of param files
    # since some nmda weights should be 0 while others > 0
    check_evoked_synkeys(p, nprox, ndist)

    # Create proximal evoked response parameters
    # f_input needs to be defined as 0
    for i in range(nprox):
        skey = 'evprox_' + str(i + 1)
        p_unique['evprox' + str(i + 1)] = {
            't0': p['t_' + skey],
            'L2_pyramidal': (p['gbar_' + skey + '_L2Pyr_ampa'],
                             p['gbar_' + skey + '_L2Pyr_nmda'],
                             0.1, p['sigma_t_' + skey]),
            'L2_basket': (p['gbar_' + skey + '_L2Basket_ampa'],
                          p['gbar_' + skey + '_L2Basket_nmda'],
                          0.1, p['sigma_t_' + skey]),
            'L5_pyramidal': (p['gbar_' + skey + '_L5Pyr_ampa'],
                             p['gbar_' + skey + '_L5Pyr_nmda'],
                             1., p['sigma_t_' + skey]),
            'L5_basket': (p['gbar_' + skey + '_L5Basket_ampa'],
                          p['gbar_' + skey + '_L5Basket_nmda'],
                          1., p['sigma_t_' + skey]),
            'prng_seedcore': int(p['prng_seedcore_' + skey]),
            'lamtha_space': 3.,
            'loc': 'proximal',
            'sync_evinput': p['sync_evinput'],
            'threshold': p['threshold'],
            'numspikes': p['numspikes_' + skey]
        }

    # Create distal evoked response parameters
    # f_input needs to be defined as 0
    for i in range(ndist):
        skey = 'evdist_' + str(i + 1)
        p_unique['evdist' + str(i + 1)] = {
            't0': p['t_' + skey],
            'L2_pyramidal': (p['gbar_' + skey + '_L2Pyr_ampa'],
                             p['gbar_' + skey + '_L2Pyr_nmda'],
                             0.1, p['sigma_t_' + skey]),
            'L5_pyramidal': (p['gbar_' + skey + '_L5Pyr_ampa'],
                             p['gbar_' + skey + '_L5Pyr_nmda'],
                             0.1, p['sigma_t_' + skey]),
            'L2_basket': (p['gbar_' + skey + '_L2Basket_ampa'],
                          p['gbar_' + skey + '_L2Basket_nmda'],
                          0.1, p['sigma_t_' + skey]),
            'prng_seedcore': int(p['prng_seedcore_' + skey]),
            'lamtha_space': 3.,
            'loc': 'distal',
            'sync_evinput': p['sync_evinput'],
            'threshold': p['threshold'],
            'numspikes': p['numspikes_' + skey]
        }

    # this needs to create many feeds
    # (amplitude, delay, mu, sigma). ordered this way to preserve compatibility
    # NEW: note double weight specification since only use ampa for gauss
    # inputs
    p_unique['extgauss'] = {
        'stim': 'gaussian',
        'L2_basket': (p['L2Basket_Gauss_A_weight'],
                      p['L2Basket_Gauss_A_weight'],
                      1., p['L2Basket_Gauss_mu'],
                      p['L2Basket_Gauss_sigma']),
        'L2_pyramidal': (p['L2Pyr_Gauss_A_weight'],
                         p['L2Pyr_Gauss_A_weight'],
                         0.1, p['L2Pyr_Gauss_mu'], p['L2Pyr_Gauss_sigma']),
        'L5_basket': (p['L5Basket_Gauss_A_weight'],
                      p['L5Basket_Gauss_A_weight'],
                      1., p['L5Basket_Gauss_mu'], p['L5Basket_Gauss_sigma']),
        'L5_pyramidal': (p['L5Pyr_Gauss_A_weight'],
                         p['L5Pyr_Gauss_A_weight'],
                         1., p['L5Pyr_Gauss_mu'], p['L5Pyr_Gauss_sigma']),
        'lamtha': 100.,
        'prng_seedcore': int(p['prng_seedcore_extgauss']),
        'loc': 'proximal',
        'threshold': p['threshold']
    }

    check_pois_synkeys(p)

    # define T_pois as 0 or -1 to reset automatically to tstop
    if p['T_pois'] in (0, -1):
        p['T_pois'] = tstop

    # Poisson distributed inputs to proximal
    # NEW: setting up AMPA and NMDA for Poisson inputs; why delays differ?
    p_unique['extpois'] = {
        'stim': 'poisson',
        'L2_basket': (p['L2Basket_Pois_A_weight_ampa'],
                      p['L2Basket_Pois_A_weight_nmda'],
                      1., p['L2Basket_Pois_lamtha']),
        'L2_pyramidal': (p['L2Pyr_Pois_A_weight_ampa'],
                         p['L2Pyr_Pois_A_weight_nmda'],
                         0.1, p['L2Pyr_Pois_lamtha']),
        'L5_basket': (p['L5Basket_Pois_A_weight_ampa'],
                      p['L5Basket_Pois_A_weight_nmda'],
                      1., p['L5Basket_Pois_lamtha']),
        'L5_pyramidal': (p['L5Pyr_Pois_A_weight_ampa'],
                         p['L5Pyr_Pois_A_weight_nmda'],
                         1., p['L5Pyr_Pois_lamtha']),
        'lamtha_space': 100.,
        'prng_seedcore': int(p['prng_seedcore_extpois']),
        't_interval': (p['t0_pois'], p['T_pois']),
        'loc': 'proximal',
        'threshold': p['threshold']
    }

    return p_ext, p_unique


# Takes two dictionaries (d1 and d2) and compares the keys in d1 to those in d2
# if any match, updates the (key, value) pair of d1 to match that of d2
# not real happy with variable names, but will have to do for now
def compare_dictionaries(d1, d2):
    # iterate over intersection of key sets (i.e. any common keys)
    for key in d1.keys() and d2.keys():
        # update d1 to have same (key, value) pair as d2
        d1[key] = d2[key]

    return d1


# debug test function
if __name__ == '__main__':
    fparam = 'param/debug.param'
