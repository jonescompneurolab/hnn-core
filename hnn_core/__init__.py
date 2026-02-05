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

__version__ = "0.5.1.dev0"


import json
from pathlib import Path
from textwrap import dedent


def _print_survey_link():
    """Print the survey link, unless the "seen" file already exists."""
    storage_dir = Path(__file__).parent
    storage_file = storage_dir / "survey_seen.json"

    if not storage_file.exists():
        print(
            dedent("""
        -------------------------------------------------------------------------------------------------------
        Thank you for installing HNN-Core! Please fill out our survey at:

            https://docs.google.com/forms/d/e/1FAIpQLSfN2F4IkGATs6cy1QBO78C6QJqvm9y14TqsCUsuR4Rrkmr1Mg/viewform

        Filling out our survey REALLY helps us to provide support and maintenance for HNN-Core.

        This message should only display once, after you have first installed HNN-Core. Happy modeling!
        -------------------------------------------------------------------------------------------------------
        """)
        )
        with open(storage_file, "w") as f:
            json.dump({"survey_seen": True}, f)


_print_survey_link()
