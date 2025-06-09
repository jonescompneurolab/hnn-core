#!/usr/bin/env python

from pathlib import Path
from hnn_core.params import convert_to_json


if __name__ == "__main__":
    top_level_dir = Path(__file__).parents[1]

    network_configs_path = Path(top_level_dir, "hnn_core", "param")
    # This is the "flat JSON" parameter file which all of our "hierarchical JSON"
    # network files are based off of
    input_flat_base_network_config_file = network_configs_path.joinpath("default.json")
    # This is the "hierarchical JSON" network file which we will build
    output_hier_base_network_config_file = network_configs_path.joinpath("jones2009_base.json")

    print(f"""
    Note that this will OVERWRITE the current contents of
    '{output_hier_base_network_config_file}'
    with a fresh generation of the base network!
    Only use this if you know what you are doing.
    """)

    convert_to_json(
        input_flat_base_network_config_file,
        output_hier_base_network_config_file,
    )
