#!/usr/bin/env python

from pathlib import Path
from shutil import copy

import hnn_core
from hnn_core.params import convert_to_json


hnn_core_root = Path(hnn_core.__file__).parent

network_configs_path = hnn_core_root / "param"
# This is the "flat JSON" parameter file which all of our "hierarchical JSON"
# network files are based off of
input_flat_base_network_config_path = network_configs_path / "default.json"
# This is the "hierarchical JSON" network file which we will build
output_hier_base_network_config_path = network_configs_path / "neymotin2020_base.json"

backup_path = Path(
    output_hier_base_network_config_path.parent,
    (
        output_hier_base_network_config_path.stem
        + "-pre_regen_BACKUP"
        + output_hier_base_network_config_path.suffix
    ),
)

print(f"""
Note that this will OVERWRITE the current contents of

'{output_hier_base_network_config_path}'

with a fresh generation of the base network! Only use this if you know what you are
doing. This will make a backup located at

'{backup_path}'.

""")

copy(output_hier_base_network_config_path, backup_path)

convert_to_json(
    input_flat_base_network_config_path,
    output_hier_base_network_config_path,
)
