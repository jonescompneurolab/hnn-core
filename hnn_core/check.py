"""Input check functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

from .params import _long_name
from .externals.mne import _validate_type, _check_option


def _check_gids(gids, gid_ranges, valid_cells, arg_name, same_type=True):
    """Format different gid specifications into list of gids"""
    _validate_type(
        gids, (int, list, range, str, None), arg_name, "int list, range, str, or None"
    )

    # Convert gids to list
    if gids is None:
        return list()
    if isinstance(gids, int):
        gids = [gids]
    elif isinstance(gids, str):
        _check_option(arg_name, gids, valid_cells)
        # Check if the name exists directly in gid_ranges first
        if gids in gid_ranges:
            gids = gid_ranges[gids]
        # If not, try converting to long name (for backward compatibility)
        elif _long_name(gids) in gid_ranges:
            gids = gid_ranges[_long_name(gids)]
        else:
            raise KeyError(f"{arg_name} '{gids}' not found in gid_ranges")

    if all(isinstance(gid, str) for gid in gids):
        processed_gids = []
        for cell_type in gids:
            if cell_type in gid_ranges:
                processed_gids.extend(gid_ranges[cell_type])
            elif _long_name(cell_type) in gid_ranges:
                processed_gids.extend(gid_ranges[_long_name(cell_type)])
            else:
                raise KeyError(f"Cell type '{cell_type}' not found in gid_ranges")
        gids = processed_gids

    cell_type = _gid_to_type(gids[0], gid_ranges)
    for gid in gids:
        _validate_type(gid, int, arg_name)
        gid_type = _gid_to_type(gid, gid_ranges)
        if gid_type is None:
            raise AssertionError(f"{arg_name} {gid} not in net.gid_ranges")
        if same_type and gid_type != cell_type:
            raise AssertionError(f"All {arg_name} must be of the same type")

    return gids


def _gid_to_type(gid, gid_ranges):
    """Reverse lookup of gid to type."""
    for gidtype, gids in gid_ranges.items():
        if gid in gids:
            return gidtype


def _string_input_to_list(input_str, valid_str, arg_name):
    """Convert input strings to list"""
    if input_str is None:
        input_str = list()
    elif isinstance(input_str, str):
        input_str = [input_str]
    for str_item in input_str:
        _check_option(arg_name, str_item, valid_str)

    return input_str
