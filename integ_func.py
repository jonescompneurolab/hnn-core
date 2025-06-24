from hnn_core.network_models import layer_separated_model, custom_geometry_model
import numpy as np
'''
Showing examples to showcase custom positions
'''
# A network with custom layer separation
net1 = layer_separated_model(
    n_pyr_x=8,
    n_pyr_y=8,
    layer_separation=1500.0,  # 1.5 mm separation
    inplane_distance=2.0      # 2 um between cells
)

weights_ampa = {'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01}
net1.add_evoked_drive('evprox1', mu=40, sigma=5, numspikes=1,
                     weights_ampa=weights_ampa, location='proximal')
net1.plot_cells()

# A now do you want to arrange cells according to a predefined relation or function you have envision
#now let this function be a hypothetical arrangement requiring circular arrangement, unlikely, but this showcases the scope 
def circular_positions(n_inner=8, n_outer=16, inner_radius=50, outer_radius=100):
    inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
    outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    
    positions = {
        'L5_pyramidal': [(outer_radius*np.cos(a), outer_radius*np.sin(a), 0) 
                         for a in outer_angles],
        'L2_pyramidal': [(outer_radius*np.cos(a), outer_radius*np.sin(a), 1300) 
                         for a in outer_angles],
        'L5_basket': [(inner_radius*np.cos(a), inner_radius*np.sin(a), 200) 
                      for a in inner_angles[:4]],
        'L2_basket': [(inner_radius*np.cos(a), inner_radius*np.sin(a), 1100) 
                      for a in inner_angles[:4]],
        'origin': (0, 0, 650)
    }
    return positions

positions = circular_positions()
net2 = custom_geometry_model(positions)
net2.plot_cells()