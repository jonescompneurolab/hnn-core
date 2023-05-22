"""
================================
XX. Modifying local connectivity
================================

This example demonstrates how to animate HNN simulations
"""

# Author: Nick Tolley <nicholas_tolley@brown.edu>

# sphinx_gallery_thumbnail_number = 2


###############################################################################
def plot_network(net, ax, t_idx, colormap):
    """
    colormap : str
        The name of a matplotlib colormap. Default: 'viridis'
    """

    if ax is None:
        ax = plt.axes(projection='3d')

    xlim = (-200, 3100)
    ylim = (-200, 3100)
    # ylim = (-3000, 3100)

    zlim = (-300, 2200)
    #viridis = cm.get_cmap('viridis', 8)

    for cell_type in net.cell_types:
        gid_range = net.gid_ranges[cell_type]
        for gid_idx, gid in enumerate(gid_range):
            print(gid, end=' ')

            cell = net.cell_types[cell_type]
            # vsec = {sec_name: ((np.array(net.cell_response.vsec[0][gid][
            #         sec_name]) - vmin) / (vmax - vmin)) for
            #         sec_name in cell.sections.keys()}
            # section_colors = {sec_name: viridis(vsec[sec_name][t_idx]) for
            #                   sec_name in cell.sections.keys()}

            section_colors = 'C0'

            pos = net.pos_dict[cell_type][gid_idx]
            pos = (float(pos[0]), float(pos[2]), float(pos[1]))
            # plot_cell_morphology(
            #     cell, ax=ax, show=False, pos=pos,
            #     xlim=xlim, ylim=ylim, zlim=zlim, color=section_colors)
            cell.plot_morphology(ax=ax, show=False, color=section_colors,
                                 pos=pos, xlim=xlim, ylim=ylim, zlim=zlim)
    # ax.view_init(10, -100)
    ax.view_init(10, -500)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    # ax.axis('on')

    return ax

def get_colors(net, t_idx, colormap):
    color_list = list()
    for cell_type in net.cell_types:
        gid_range = net.gid_ranges[cell_type]
        for gid_idx, gid in enumerate(gid_range):

            cell = net.cell_types[cell_type]
            vmin, vmax = -100, 50

            for sec_name in cell.sections.keys():
                vsec = (np.array(net.cell_response.vsec[0][gid][sec_name]) - vmin) / (vmax - vmin)
                color_list.append(colormap(vsec[t_idx]))
    return color_list

                                

def update_colors(ax, net, t_idx, colormap):
    color_list = get_colors(net, t_idx, colormap)
    lines = ax.get_lines()
    for line, color in zip(lines, color_list):
        line.set_color(color)
    ax.view_init(10, -500)
    

net = jones_2009_model()
net.set_cell_positions(inplane_distance=300)
add_erp_drives_to_jones_model(net)
dpl = simulate_dipole(net, dt=0.5, tstop=100, record_vsec='all')


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_network(net, ax=ax, t_idx=None, colormap=None)

colormap = colormaps['viridis']
update_colors(ax, net, t_idx=100, colormap=colormap)
ax.view_init(20, 100)
fig
