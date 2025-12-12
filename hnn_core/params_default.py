"""Master list of changeable params."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>


def get_params_default(nprox=2, ndist=1):
    """ Note that nearly all start times are set BEYOND tstop for this file
        Most values here are set to whatever default value
        inactivates them, such as 0 for conductance
        prng seed values are also set to 0 (non-random)
        flat file of default values
        will most often be overwritten
    """
    # set default params
    p = {
        'sim_prefix': 'default',

        # simulation end time (ms)
        'tstop': 250.,

        # numbers of cells making up the pyramidal grids
        'N_pyr_x': 1,
        'N_pyr_y': 1,

        # amplitudes of individual Gaussian random inputs to L2Pyr and L5Pyr
        # L2 Basket params
        'L2Basket_Gauss_A_weight': 0.,
        'L2Basket_Gauss_mu': 2000.,
        'L2Basket_Gauss_sigma': 3.6,
        'L2Basket_Pois_A_weight_ampa': 0.,
        'L2Basket_Pois_A_weight_nmda': 0.,
        'L2Basket_Pois_lamtha': 0.,

        # L2 Pyr params
        'L2Pyr_Gauss_A_weight': 0.,
        'L2Pyr_Gauss_mu': 2000.,
        'L2Pyr_Gauss_sigma': 3.6,
        'L2Pyr_Pois_A_weight_ampa': 0.,
        'L2Pyr_Pois_A_weight_nmda': 0.,
        'L2Pyr_Pois_lamtha': 0.,

        # L5 Pyr params
        'L5Pyr_Gauss_A_weight': 0.,
        'L5Pyr_Gauss_mu': 2000.,
        'L5Pyr_Gauss_sigma': 4.8,
        'L5Pyr_Pois_A_weight_ampa': 0.,
        'L5Pyr_Pois_A_weight_nmda': 0.,
        'L5Pyr_Pois_lamtha': 0.,

        # L5 Basket params
        'L5Basket_Gauss_A_weight': 0.,
        'L5Basket_Gauss_mu': 2000.,
        'L5Basket_Gauss_sigma': 2.,
        'L5Basket_Pois_A_weight_ampa': 0.,
        'L5Basket_Pois_A_weight_nmda': 0.,
        'L5Basket_Pois_lamtha': 0.,

        # maximal conductances for all synapses
        # max conductances TO L2Pyrs
        'gbar_L2Pyr_L2Pyr_ampa': 0.,
        'gbar_L2Pyr_L2Pyr_nmda': 0.,
        'gbar_L2Basket_L2Pyr_gabaa': 0.,
        'gbar_L2Basket_L2Pyr_gabab': 0.,

        # max conductances TO L2Baskets
        'gbar_L2Pyr_L2Basket': 0.,
        'gbar_L2Basket_L2Basket': 0.,

        # max conductances TO L5Pyr
        'gbar_L5Pyr_L5Pyr_ampa': 0.,
        'gbar_L5Pyr_L5Pyr_nmda': 0.,
        'gbar_L2Pyr_L5Pyr': 0.,
        'gbar_L2Basket_L5Pyr': 0.,
        'gbar_L5Basket_L5Pyr_gabaa': 0.,
        'gbar_L5Basket_L5Pyr_gabab': 0.,

        # max conductances TO L5Baskets
        'gbar_L5Basket_L5Basket': 0.,
        'gbar_L5Pyr_L5Basket': 0.,
        'gbar_L2Pyr_L5Basket': 0.,

        # Ongoing proximal alpha rhythm
        'distribution_prox': 'normal',
        't0_input_prox': 1000.,
        'tstop_input_prox': 250.,
        'f_input_prox': 10.,
        'f_stdev_prox': 20.,
        'events_per_cycle_prox': 2,
        'repeats_prox': 10,
        't0_input_stdev_prox': 0.0,

        # Ongoing distal alpha rhythm
        'distribution_dist': 'normal',
        't0_input_dist': 1000.,
        'tstop_input_dist': 250.,
        'f_input_dist': 10.,
        'f_stdev_dist': 20.,
        'events_per_cycle_dist': 2,
        'repeats_dist': 10,
        't0_input_stdev_dist': 0.0,

        # thalamic input amplitudes and delays
        'input_prox_A_weight_L2Pyr_ampa': 0.,
        'input_prox_A_weight_L2Pyr_nmda': 0.,
        'input_prox_A_weight_L5Pyr_ampa': 0.,
        'input_prox_A_weight_L5Pyr_nmda': 0.,
        'input_prox_A_weight_L2Basket_ampa': 0.,
        'input_prox_A_weight_L2Basket_nmda': 0.,
        'input_prox_A_weight_L5Basket_ampa': 0.,
        'input_prox_A_weight_L5Basket_nmda': 0.,
        'input_prox_A_delay_L2': 0.1,
        'input_prox_A_delay_L5': 1.0,

        # current values, not sure where these distal values come from, need to
        # check
        'input_dist_A_weight_L2Pyr_ampa': 0.,
        'input_dist_A_weight_L2Pyr_nmda': 0.,
        'input_dist_A_weight_L5Pyr_ampa': 0.,
        'input_dist_A_weight_L5Pyr_nmda': 0.,
        'input_dist_A_weight_L2Basket_ampa': 0.,
        'input_dist_A_weight_L2Basket_nmda': 0.,
        'input_dist_A_delay_L2': 5.,
        'input_dist_A_delay_L5': 5.,

        # times and stdevs for evoked responses
        'dt_evprox0_evdist': -1,  # not used in GUI
        'dt_evprox0_evprox1': -1,  # not used in GUI
        # whether evoked inputs arrive at same time to all cells
        'sync_evinput': 1,
        # increment (ms) for avg evoked input start (for trial n, avg start
        # time is n * evinputinc
        'inc_evinput': 0.0,

        # analysis
        'save_spec_data': 0,
        'f_max_spec': 40.,
        'spec_cmap': 'jet',  # only used in GUI
        'dipole_scalefctr': 30e3,  # scale factor for dipole - default at 30e3
        # based on scaling needed to match model ongoing rhythms from
        # jones 2009 - for ERPs can use 300
        # for ongoing rhythms + ERPs ... use ... ?
        # window for smoothing (box filter) - 15 ms from jones 2009; shorten
        'dipole_smooth_win': 15.0,
        # in case want to look at higher frequency activity
        'save_figs': 0,
        'save_dpl': 0,  # whether to write dipole output to a file
        'record_vsoma': 0,  # whether to record somatic voltages
        'record_isoma': 0,  # whether to record somatic currents
        'record_vsec': 0,  # whether to record voltages
        'record_isec': 0,  # whether to record currents
        'record_ca': 0,  # whether to record calcium concentration

        # numerics
        # N_trials of 1 means that seed is set by rank
        'N_trials': 1,

        # prng_state is a string for a filename containing the
        # random state one wants to use
        # prng seed cores are the base integer seed for the specific
        # prng object for a specific random number stream
        # 'prng_state': None,
        'prng_seedcore_input_prox': 0,
        'prng_seedcore_input_dist': 0,
        'prng_seedcore_extpois': 0,
        'prng_seedcore_extgauss': 0,

        # default end time for pois inputs
        't0_pois': 0.,
        'T_pois': -1,
        'dt': 0.025,
        'celsius': 37.0,
        'threshold': 0.0  # firing threshold
    }

    # grab cell-specific params and update p accordingly
    p_L2Pyr = get_L2Pyr_params_default()
    p_L5Pyr = get_L5Pyr_params_default()
    p.update(p_L2Pyr)
    p.update(p_L5Pyr)

    # get evoked params and update p accordingly
    p_ev_prox = get_ev_params_default(nprox, True)
    p_ev_dist = get_ev_params_default(ndist, False)
    p.update(p_ev_prox)
    p.update(p_ev_dist)

    return p

# return dict with default params (empty) for evoked inputs;
# n is number of evoked inputs
# isprox == True iff proximal (otherwise distal)


def get_ev_params_default(n, isprox):
    dout = {}  # OrderedDict()
    if isprox:
        pref = 'evprox'
    else:
        pref = 'evdist'
    lty = ['L2Pyr', 'L5Pyr', 'L2Basket']
    if isprox:
        lty.append('L5Basket')
    lsy = ['ampa', 'nmda']  # allow changing both ampa and nmda weights
    for i in range(n):
        tystr = pref + '_' + str(i + 1)  # this string includes input number
        for ty in lty:
            for sy in lsy:
                dout['gbar_' + tystr + '_' + ty +
                     '_' + sy] = 0.  # feed strength
        dout['t_' + tystr] = 0.  # times and stdevs for evoked responses
        dout['sigma_t_' + tystr] = 0.
        # random number generator seed for this input
        dout['prng_seedcore_' + tystr] = 0
        # number of presynaptic spikes (postsynaptic inputs)
        dout['numspikes_' + tystr] = 1
    return dout


def get_L2Pyr_params_default():
    """Returns default params for L2 pyramidal cell."""
    return {
        # Soma
        'L2Pyr_soma_L': 22.1,
        'L2Pyr_soma_diam': 23.4,
        'L2Pyr_soma_cm': 1,
        'L2Pyr_soma_Ra': 100.,

        # Dendrites
        'L2Pyr_dend_cm': 1,
        'L2Pyr_dend_Ra': 100.,

        'L2Pyr_apicaltrunk_L': 59.5,
        'L2Pyr_apicaltrunk_diam': 4.25,

        'L2Pyr_apical1_L': 306.,
        'L2Pyr_apical1_diam': 4.08,

        'L2Pyr_apicaltuft_L': 238.,
        'L2Pyr_apicaltuft_diam': 3.4,

        'L2Pyr_apicaloblique_L': 340.,
        'L2Pyr_apicaloblique_diam': 3.91,

        'L2Pyr_basal1_L': 85.,
        'L2Pyr_basal1_diam': 4.25,

        'L2Pyr_basal2_L': 255.,
        'L2Pyr_basal2_diam': 2.72,

        'L2Pyr_basal3_L': 255.,
        'L2Pyr_basal3_diam': 2.72,

        # Synapses
        'L2Pyr_ampa_e': 0.,
        'L2Pyr_ampa_tau1': 0.5,
        'L2Pyr_ampa_tau2': 5.,
        'L2Pyr_ampa_type': "Exp2Syn",

        'L2Pyr_nmda_e': 0.,
        'L2Pyr_nmda_tau1': 1.,
        'L2Pyr_nmda_tau2': 20.,
        'L2Pyr_nmda_type': "Exp2Syn",

        'L2Pyr_gabaa_e': -80.,
        'L2Pyr_gabaa_tau1': 0.5,
        'L2Pyr_gabaa_tau2': 5.,
        'L2Pyr_gabaa_type': "Exp2Syn",

        'L2Pyr_gabab_e': -80.,
        'L2Pyr_gabab_tau1': 1.,
        'L2Pyr_gabab_tau2': 20.,
        'L2Pyr_gabab_type': "Exp2Syn",

        # Biophysics soma
        'L2Pyr_soma_gkbar_hh2': 0.01,
        'L2Pyr_soma_gnabar_hh2': 0.18,
        'L2Pyr_soma_el_hh2': -65.,
        'L2Pyr_soma_gl_hh2': 4.26e-5,
        'L2Pyr_soma_gbar_km': 250.,

        # Biophysics dends
        'L2Pyr_dend_gkbar_hh2': 0.01,
        'L2Pyr_dend_gnabar_hh2': 0.15,
        'L2Pyr_dend_el_hh2': -65.,
        'L2Pyr_dend_gl_hh2': 4.26e-5,
        'L2Pyr_dend_gbar_km': 250.,
    }


def get_L5Pyr_params_default():
    """Returns default params for L5 pyramidal cell."""
    return {
        # Soma
        'L5Pyr_soma_L': 39.,
        'L5Pyr_soma_diam': 28.9,
        'L5Pyr_soma_cm': 0.85,
        'L5Pyr_soma_Ra': 200.,

        # Dendrites
        'L5Pyr_dend_cm': 0.85,
        'L5Pyr_dend_Ra': 200.,

        'L5Pyr_apicaltrunk_L': 102.,
        'L5Pyr_apicaltrunk_diam': 10.2,

        'L5Pyr_apical1_L': 680.,
        'L5Pyr_apical1_diam': 7.48,

        'L5Pyr_apical2_L': 680.,
        'L5Pyr_apical2_diam': 4.93,

        'L5Pyr_apicaltuft_L': 425.,
        'L5Pyr_apicaltuft_diam': 3.4,

        'L5Pyr_apicaloblique_L': 255.,
        'L5Pyr_apicaloblique_diam': 5.1,

        'L5Pyr_basal1_L': 85.,
        'L5Pyr_basal1_diam': 6.8,

        'L5Pyr_basal2_L': 255.,
        'L5Pyr_basal2_diam': 8.5,

        'L5Pyr_basal3_L': 255.,
        'L5Pyr_basal3_diam': 8.5,

        # Synapses
        'L5Pyr_ampa_e': 0.,
        'L5Pyr_ampa_tau1': 0.5,
        'L5Pyr_ampa_tau2': 5.,
        'L5Pyr_ampa_type': "Exp2Syn",

        'L5Pyr_nmda_e': 0.,
        'L5Pyr_nmda_tau1': 1.,
        'L5Pyr_nmda_tau2': 20.,
        'L5Pyr_nmda_type': "Exp2Syn",

        'L5Pyr_gabaa_e': -80.,
        'L5Pyr_gabaa_tau1': 0.5,
        'L5Pyr_gabaa_tau2': 5.,
        'L5Pyr_gabaa_type': "Exp2Syn",

        'L5Pyr_gabab_e': -80.,
        'L5Pyr_gabab_tau1': 1.,
        'L5Pyr_gabab_tau2': 20.,
        'L5Pyr_gabab_type': "Exp2Syn",

        # Biophysics soma
        'L5Pyr_soma_gkbar_hh2': 0.01,
        'L5Pyr_soma_gnabar_hh2': 0.16,
        'L5Pyr_soma_el_hh2': -65.,
        'L5Pyr_soma_gl_hh2': 4.26e-5,
        'L5Pyr_soma_gbar_ca': 60.,
        'L5Pyr_soma_taur_cad': 20.,
        'L5Pyr_soma_gbar_kca': 2e-4,
        'L5Pyr_soma_gbar_km': 200.,
        'L5Pyr_soma_gbar_cat': 2e-4,
        'L5Pyr_soma_gbar_ar': 1e-6,

        # Biophysics dends
        'L5Pyr_dend_gkbar_hh2': 0.01,
        'L5Pyr_dend_gnabar_hh2': 0.14,
        'L5Pyr_dend_el_hh2': -71.,
        'L5Pyr_dend_gl_hh2': 4.26e-5,
        'L5Pyr_dend_gbar_ca': 60.,
        'L5Pyr_dend_taur_cad': 20.,
        'L5Pyr_dend_gbar_kca': 2e-4,
        'L5Pyr_dend_gbar_km': 200.,
        'L5Pyr_dend_gbar_cat': 2e-4,
        'L5Pyr_dend_gbar_ar': 1e-6,
    }
def get_L2Pyrhuman_params():

    return {# Soma
        'L2Pyr_soma_L': 22.1,
        'L2Pyr_soma_diam': 23.4,
        'L2Pyr_soma_cm': 1.5,
        'L2Pyr_soma_Ra': 200.,

        # Dendrites
        'L2Pyr_dend_cm': 1.5,
        'L2Pyr_dend_Ra': 200.,

        'L2Pyr_apicaltrunk_L': 59.5,
        'L2Pyr_apicaltrunk_diam': 4.25,

        'L2Pyr_apical1_L': 306.,
        'L2Pyr_apical1_diam': 4.08,

        'L2Pyr_apicaltuft_L': 238.,
        'L2Pyr_apicaltuft_diam': 3.4,

        'L2Pyr_apicaloblique_L': 340.,
        'L2Pyr_apicaloblique_diam': 3.91,

        'L2Pyr_basal1_L': 85.,
        'L2Pyr_basal1_diam': 4.25,

        'L2Pyr_basal2_L': 255.,
        'L2Pyr_basal2_diam': 2.72,

        'L2Pyr_basal3_L': 255.,
        'L2Pyr_basal3_diam': 2.72,
        
        # Synapses
        'L2Pyr_ampa_e': 0.,
        'L2Pyr_ampa_tau1': 0.5,
        'L2Pyr_ampa_tau2': 5.,
        'L2Pyr_ampa_type': "Exp2Syn",

        'L2Pyr_nmda_e': 0.,
        'L2Pyr_nmda_tau1': 15.,
        'L2Pyr_nmda_tau2': 150.,
        'L2Pyr_nmda_type': "NMDAeee_KD",

        'L2Pyr_gabaa_e': -80.,
        'L2Pyr_gabaa_tau1': 0.5,
        'L2Pyr_gabaa_tau2': 5.,
        'L2Pyr_gabaa_type': "Exp2Syn",

        'L2Pyr_gabab_e': -80.,
        'L2Pyr_gabab_tau1': 45.,
        'L2Pyr_gabab_tau2': 200.,
        'L2Pyr_gabab_type': "gabab",

        # Biophysics soma
        'L2Pyr_soma_gbar_NaTs2_t_32d': 20_400e-4*1.2,
        'L2Pyr_soma_gbar_SKv3_1': 6_930e-4,
        'L2Pyr_soma_gbar_Nap_Et2': 1e-4/2*0,
        'L2Pyr_soma_gbar_SK_E2': 3.e-05,
        'L2Pyr_soma_gbar_Ca_HVA': 9.92e-04,
        'L2Pyr_soma_gbar_Ca_LVAst': 34.3e-4,
        'L2Pyr_soma_gbar_Ih': 0.000080/2,
        'L2Pyr_soma_gbar_Im': 8e-3*0.05,
        'L2Pyr_soma_g_pas': 1.0 / 15000,
        'L2Pyr_soma_e_pas': -75,
        'L2Pyr_soma_decay_CaDynamics_E2' : 90,
        'L2Pyr_soma_gamma_CaDynamics_E2' : 0.000533,

        # Biophysics basal
        'L2Pyr_basal_gbar_NaTs2_t_32d': 0.008009,
        'L2Pyr_basal_gbar_SKv3_1': 0.000513,
        'L2Pyr_basal_gbar_Ih': 0.00008/2,
        'L2Pyr_basal_g_pas': 1.0 / 13000.0,         # default read out from neuron
        'L2Pyr_basal_e_pas': -75, 
        'L2Pyr_basal_decay_CaDynamics_E2' : 70,
        'L2Pyr_basal_gamma_CaDynamics_E2' : 0.000533*2.5,

        # Biophysics dends
        'L2Pyr_dend_gbar_NaTa_t_32d': 0.008*0.0001,
        'L2Pyr_dend_gbar_SKv3_1': 0.003*3,
        'L2Pyr_dend_gbar_Ca_HVA': 0.00001,
        'L2Pyr_dend_gbar_Ca_LVAst': 0.0000001,
        'L2Pyr_dend_gbar_Ih': 0.000080,
        'L2Pyr_dend_gbar_Im': 0.0009*0.0001,
        'L2Pyr_dend_gbar_SK_E2': 3.e-06,
        'L2Pyr_dend_g_pas': 1.0 /9000,
        'L2Pyr_dend_e_pas': -75,
        'L2Pyr_dend_decay_CaDynamics_E2' : 50,
        'L2Pyr_dend_gamma_CaDynamics_E2' : .0005096}

def get_L5PyrET_params():

    return {
        # Soma
        'L5Pyr_soma_L': 39.,
        'L5Pyr_soma_diam': 28.9,
        'L5Pyr_soma_cm': 1,
        'L5Pyr_soma_Ra': 100, #Rich 495.73, Hay: 100

        # Dendrite
        'L5Pyr_dend_cm': 1,
        'L5Pyr_dend_Ra': 100, #Rich 495.73, Hay: 100
        'L5Pyr_basal_cm': 1,
        'L5Pyr_basal_Ra': 100, #Rich 495.73, Hay: 100

        'L5Pyr_apicaltrunk_L': 102.,
        'L5Pyr_apicaltrunk_diam': 10.2,

        'L5Pyr_apical1_L': 680.,
        'L5Pyr_apical1_diam': 7.48,

        'L5Pyr_apical2_L': 680.,
        'L5Pyr_apical2_diam': 4.93,

        'L5Pyr_apicaltuft_L': 425.,
        'L5Pyr_apicaltuft_diam': 3.4,

        'L5Pyr_apicaloblique_L': 255.,
        'L5Pyr_apicaloblique_diam': 5.1,

        'L5Pyr_basal1_L': 85.,
        'L5Pyr_basal1_diam': 6.8,

        'L5Pyr_basal2_L': 255.,
        'L5Pyr_basal2_diam': 8.5,

        'L5Pyr_basal3_L': 255.,
        'L5Pyr_basal3_diam': 8.5,

        # Synapses
        'L5Pyr_ampa_e': 0.,
        'L5Pyr_ampa_tau1': 0.5,
        'L5Pyr_ampa_tau2': 5.,
        'L5Pyr_ampa_type': "Exp2Syn",

        'L5Pyr_nmda_e': 0.,
        'L5Pyr_nmda_tau1': 15.,
        'L5Pyr_nmda_tau2': 150.,
        'L5Pyr_nmda_type': "NMDAeee_KD",

        'L5Pyr_gabaa_e': -80.,
        'L5Pyr_gabaa_tau1': 0.5,
        'L5Pyr_gabaa_tau2': 5.,
        'L5Pyr_gabaa_type': "Exp2Syn",

        'L5Pyr_gabab_e': -80.,
        'L5Pyr_gabab_tau1': 45,
        'L5Pyr_gabab_tau2': 200.,
        'L5Pyr_gabab_type': "gabab",

        # Biophysics soma
        'L5Pyr_soma_gbar_NaTs2_t': 20_400e-4/2*1.5,
        'L5Pyr_soma_gbar_SKv3_1': 6_930e-4/2,
        'L5Pyr_soma_gbar_Nap_Et2': 17.2e-4/2*0.11,
        'L5Pyr_soma_gbar_SK_E2': 441e-4*15,
        'L5Pyr_soma_gbar_Ca_HVA': 9.92e-04/2,
        'L5Pyr_soma_gbar_Ca_LVAst': 34.3e-4/2,
        'L5Pyr_soma_gbar_Ih': 1e-4/2,
        'L5Pyr_soma_gbar_Im': 1e-4,
        'L5Pyr_soma_gbar_K_Pst': 22.3e-4/2,
        'L5Pyr_soma_gbar_K_Tst': 812e-4/2,
        'L5Pyr_soma_g_pas': .338e-4/4,
        'L5Pyr_soma_e_pas': -90,
        'L5Pyr_soma_decay_CaDynamics_E2' : 460,
        'L5Pyr_soma_gamma_CaDynamics_E2' : .000501,#*1e-10,

        # Biophysics basal
        'L5Pyr_basal_gbar_NaTs2_t': 0.0102,
        'L5Pyr_basal_gbar_SKv3_1': 0.0001305,
        'L5Pyr_basal_gbar_Ih': 5.14e-5/4*2,
        'L5Pyr_basal_g_pas': 1.75e-5/2*3,
        'L5Pyr_basal_e_pas': -90,
        'L5Pyr_basal_decay_CaDynamics_E2' : 50,
        'L5Pyr_basal_gamma_CaDynamics_E2' : .0005096,

        # Biophysics dends
        'L5Pyr_dend_gbar_NaTa_t': 0.0213/2,
        'L5Pyr_dend_gbar_SKv3_1': 0.000261/2*2,
        'L5Pyr_dend_gbar_SK_E2': 0.0012/2*6,
        'L5Pyr_dend_gbar_Ca_HVA': 2.78e-5/2,
        'L5Pyr_dend_gbar_Ca_LVAst': 93.5e-6/2,
        'L5Pyr_dend_gbar_Ih': 1e-4/2*1.2,

        'L5Pyr_dend_gbar_Im': 0.0000675/2,
        'L5Pyr_dend_gbar_K_Pst': 0,
        'L5Pyr_dend_gbar_K_Tst': 0,
        'L5Pyr_dend_g_pas':  0.0000589/1.5,
        'L5Pyr_dend_e_pas': -85,
        'L5Pyr_dend_decay_CaDynamics_E2' : 122,
        'L5Pyr_dend_gamma_CaDynamics_E2' : .0005096*1
    }

def get_Int_params():

    return {'Int_gbar_nas': 0.1125, 
            'Int_gbar_kdr': 0.225*2,
            'Int_gbar_kd': 0.00039*2,
            'Int_gbar_Ih': 2e-5*9,
            'Int_g_pas': .0001,
            'Int_e_pas': -70,
            # track calcium dynamics for new NMDA
            'Int_decay_CaDynamics_E2' : 50,
            'Int_gamma_CaDynamics_E2' : .0005096,

            # Synapses
            'Int_ampa_e': 0.,
            # Destexhe et al., 1998: interneuron rise and decay twice as fast!
            'Int_ampa_tau1': 0.35,
            'Int_ampa_tau2': 2.5,
            'Int_ampa_type': "Exp2Syn",

            'Int_nmda_e': 0.,
            'Int_nmda_tau1': 15.,
            'Int_nmda_tau2': 150.,
            'Int_nmda_type': "NMDAeee_KD",

            'Int_gabaa_e': -80.,
            'Int_gabaa_tau1': 0.5,
            'Int_gabaa_tau2': 5.,
            'Int_gabaa_type': "Exp2Syn"}