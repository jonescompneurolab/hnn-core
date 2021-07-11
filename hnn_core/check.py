"""Input check functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

from .params import _long_name
from .externals.mne import _validate_type, _check_option


def _create_gid_list(gids, gid_ranges, valid_cells, arg_name):
    """Format different gid specifications into list of gids"""
    _validate_type(gids, (int, list, range, str, None), arg_name,
                   'int list, range, str, or None')

    # Convert gids to list
    if isinstance(gids, int):
        gids = [gids]
    elif isinstance(gids, str):
        _check_option(arg_name, gids, valid_cells)
        gids = gid_ranges[_long_name(gids)]

    cell_type = _gid_to_type(gids[0], gid_ranges)
    for gid in gids:
        _validate_type(gid, int, arg_name)
        gid_type = _gid_to_type(gid, gid_ranges)
        if gid_type is None:
            raise AssertionError(
                f'{arg_name} {gid} not in net.gid_ranges')
        if gid_type != cell_type:
            raise AssertionError(f'All {arg_name} must be of the same type')

    return gids


def _gid_to_type(gid, gid_ranges):
    """Reverse lookup of gid to type."""
    for gidtype, gids in gid_ranges.items():
        if gid in gids:
            return gidtype


def _check_gid_range(gid, gid_ranges, arg_name):
    gid_type = _gid_to_type(gid, gid_ranges)
    if gid_type is None:
        raise AssertionError(
            f'{arg_name} {gid} not in net.gid_ranges')