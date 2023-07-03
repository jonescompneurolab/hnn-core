"""Temp file to get coords"""

from hnn_core import pyramidal


def find_coords(func_name, cell_name):
    cell = func_name(cell_name=cell_name)
    # print(cell.sections['soma'].end_pts)
    for section_name, section in cell.sections.items():
        print(section_name)
        print(section.end_pts)
    cell.build()
    print(cell._nrn_sections)


find_coords(pyramidal, cell_name='L2Pyr')
