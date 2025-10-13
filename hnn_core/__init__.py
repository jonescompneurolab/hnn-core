from .dipole import (
    simulate_dipole,
    read_dipole,
    average_dipoles,
    Dipole,
    _read_dipole_txt,
)
from .params import Params, read_params, convert_to_json
from .network import Network, pick_connection
from .network_models import jones_2009_model, law_2021_model, calcium_model
from .cell import Cell
from .cell_response import CellResponse, read_spikes
from .cells_default import pyramidal, basket
from .parallel_backends import MPIBackend, JoblibBackend
from .hnn_io import (
    dict_to_network,
    network_to_dict,
    read_network_configuration,
    write_network_configuration,
)

__version__ = "0.4.4.dev0"

def _maybe_check_first_run():
    try:
        from .first_run import check_first_run
        check_first_run()
    except ImportError as e:
        print("Warning: could not run first-run check:", e)

_maybe_check_first_run()
