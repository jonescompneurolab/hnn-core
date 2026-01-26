import json
from pathlib import Path
from textwrap import dedent

import hnn_core


def _print_survey_link():
    """Print the survey link, unless the "seen" file already exists."""
    storage_dir = Path(hnn_core.__file__).parent
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
