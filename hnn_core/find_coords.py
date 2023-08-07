"""Temp file to get coords"""

from hnn_core import pyramidal
from neuron import h


def test_distance(func_name, cell_name):
    # soma = h.Section(name='soma')
    # dend = h.Section(name='dend')
    # dend.connect(soma(0.5))

    # # Define soma points
    # h.pt3dadd(-10, 0, 0, 1, sec=soma)
    # h.pt3dadd(0, 0, 0, 1, sec=soma)
    # h.pt3dadd(10, 0, 0, 1, sec=soma)

    # # Define dend points
    # h.pt3dadd(0,0,0,1, sec=dend)
    # h.pt3dadd(50,0,0,1, sec=dend)
    # h.pt3dadd(50,50,0,1, sec=dend)

    # h.distance(sec=soma)
    # print(h.distance(dend(0)))
    # print(h.distance(dend(0.5)))
    # print(h.distance(dend(1)))

    # dend2 = h.Section(name='dend')

    # dend2.connect(dend(1))
    # soma.L = 10
    # dend.L = 100
    # dend2.L = 100

    # print(h.distance(dend(0),dend2(1)))

    # h.define_shape()
    # h.distance(sec=soma)
    # for seg in dend:
    #     print(seg.x)
    #     print(h.distance(seg.x))

    # print coordinates and distance
    # for x in range(dend.n3d()):
    #     i = x/2
    #     print(dend.x3d(x))
    #     print(dend.y3d(x))
    #     print(dend.z3d(x))
    #     print(h.distance(dend(i)))
    # print(h.distance(dend(0)))
    # print(h.distance(dend(0.5)))
    # print(h.distance(dend(1)))
    # length = h.distance(soma(0.5), dend(0))
    # print(length)

    cell = func_name(cell_name=cell_name)
    cell.build()
    h.distance(sec=cell._nrn_sections['soma'])
    for sec_name, section in cell.sections.items():
        sec = cell._nrn_sections[sec_name]
        # sec.L = 1000
        print("Section length is %d" % (sec.L,))
        for seg in sec:
            print(sec_name)
            print(seg.x)
            print(h.distance(seg.x))
            print(h.distance(sec(seg.x)))

    # Notes
    # h.distance(section1(i),section2(j)) calculates the path distance
    # between the 2 points defined and not the euclidean distance.
    # h.distance(sec=sec) makes the 0 end of the section as the origin.
    # After origin is set, all paths are calculated along the section.
    # Absolute coordinates are not used (no euclidean distance)
    # h.distance(sec(x)) gives distance between the x location of section
    # to the origin set if path to the origin exists.
    # Behavior of seg.x ?


def find_coords(func_name, cell_name):
    cell = func_name(cell_name=cell_name)
    # print(cell.sections['soma'].end_pts)
    # for section_name, section in cell.sections.items():
    #     print(section_name)
    #     print(section.end_pts)
    cell.build()
    # print(cell._nrn_sections)


def test_cell_tree(func_name, cell_name):
    cell = func_name(cell_name=cell_name)
    # cell_tree = cell.cell_tree
    # for key in cell_tree.keys():
    #     print(key)
    #     print(cell_tree[key])
    cell_new = func_name(cell_name=cell_name)
    cell._update_end_pts()
    cell_new.update_end_pts()
    print("pos is ")
    print(cell_new.pos)
    print("\n")
    for sec_name in cell.sections:
        print(sec_name)
        print("old coordinates")
        print(cell.sections[sec_name].end_pts)
        print("new coordinates")
        print(cell_new.sections[sec_name].end_pts)
        print("length of section is ", cell_new.sections[sec_name].L)
        print("\n")


def test_distance_(func_name, cell_name):
    cell = func_name(cell_name=cell_name)
    cell_new = func_name(cell_name=cell_name)

    print(cell_new.sections['soma'].L)

    cell._update_end_pts()
    cell.build()

    cell_new.update_end_pts()
    cell_new.sections = cell_new._set_biophysics_new(cell_new.sections)

    # for sec_name in cell.sections.keys():
    #     print(sec_name)
    #     for mech_name in cell.sections[sec_name].mechs.keys():
    #         print(mech_name)
    #         for attr, val in cell.sections[sec_name].mechs[mech_name].items():  # noqa
    #             print(attr)
    #             print(val)
    #         for attr, val in cell_new.sections[sec_name].mechs[mech_name].items():  # noqa
    #             print(attr)
    #             print(val)


# find_coords(pyramidal, cell_name='L5Pyr')
# test_distance(pyramidal, cell_name='L2Pyr')
# test_cell_tree(pyramidal, cell_name='L2Pyr')
test_distance_(pyramidal, cell_name='L5Pyr')
