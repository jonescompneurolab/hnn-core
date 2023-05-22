"""
================================
XX. Modifying local connectivity
================================

This example demonstrates how to animate HNN simulations
"""

# Author: Nick Tolley <nicholas_tolley@brown.edu>


###############################################################################
from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.viz import NetworkPlot

net = jones_2009_model()
net.set_cell_positions(inplane_distance=300)
add_erp_drives_to_jones_model(net)
dpl = simulate_dipole(net, dt=0.5, tstop=170, record_vsec='all')

net_plot = NetworkPlot(net)

###############################################################################
from ipywidgets import interact, IntSlider

def update_plot(t_idx, elev, azim):
    net_plot.update_section_voltages(t_idx)
    net_plot.elev = elev
    net_plot.azim = azim
    return net_plot.fig

time_slider = IntSlider(min=0, max=len(net_plot.times), value=1, continuous_update=False)
elev_slider = IntSlider(min=-100, max=100, value=10, continuous_update=False)
azim_slider = IntSlider(min=-100, max=100, value=-100, continuous_update=False)

interact(update_plot, t_idx=time_slider, elev=elev_slider, azim=azim_slider)
