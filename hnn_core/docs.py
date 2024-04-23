"""The documentation functions."""

docdict = dict()
# Define docdicts

docdict[
    "net"
] = """
net : Instance of Network object
    The Network object.
"""

docdict[
    "fname"
] = """
fname : str | Path object
    Full path to the output file (.hdf5).
"""

docdict[
    "overwrite"
] = """
overwrite : Boolean
    True : Overwrite existing file.
    False : Throw error if file already exists.
"""

docdict[
    "write_output"
] = """
write_output : Boolean
    True : Save the Network simulation output.
    False : Do not save the Network simulation output.
"""

docdict[
    "read_output"
] = """
read_output : Boolean
    True : Read network with simulation results.
    False : Read network without simulation results.
"""

docdict[
    "read_drives"
] = """
read_output : Boolean
    True : Read drives from configuration file.
    False : Do not read drives from the configuration file.
"""
