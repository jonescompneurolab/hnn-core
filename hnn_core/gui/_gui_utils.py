# clear empty trash simulations
def clear_empty_trash_simualtions(simulation_data):

    # AES: a "trash" simulation appears to be created (named "default") even if
    # all a user does is load an external dipole data file. However, I do not
    # fully understand how VizManager et al. manages the simulation data (I find
    # it very confusing) so I am NOT touching it.
    for _name in tuple(simulation_data.keys()):
        if not simulation_data[_name]["dpls"]:
            del simulation_data[_name]
