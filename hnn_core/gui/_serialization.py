"""Serialization of simulation data and network configuration to downloadable formats."""

import io
import zipfile

import numpy as np

from hnn_core.hnn_io import write_network_configuration


def serialize_simulation(simulations_data, simulation_name):
    """Serializes simulation data to CSV.

    Creates a single CSV file or a ZIP file containing multiple CSVs,
    depending on the number of trials in the simulation.

    """
    csv_trials_output = []
    # CSV file headers
    headers = "times,agg,L2,L5"
    fmt = "%f, %f, %f, %f"

    ## retrieve simulation by name
    simulation_data = simulations_data[simulation_name]
    for dpl_trial in simulation_data["dpls"]:
        # Combine all data columns at once
        signals_matrix = np.column_stack(
            (
                dpl_trial.times,
                dpl_trial.data["agg"],
                dpl_trial.data["L2"],
                dpl_trial.data["L5"],
            )
        )

        # Using StringIO to collect CSV data
        with io.StringIO() as output:
            np.savetxt(output, signals_matrix, delimiter=",", header=headers, fmt=fmt)
            csv_trials_output.append(output.getvalue())

    if len(csv_trials_output) == 1:
        # Return a single csv file
        return csv_trials_output[0], ".csv"
    else:
        # Create zip file
        return _create_zip(csv_trials_output, simulation_name), ".zip"


def serialize_config(simulations_data, simulation_name):
    """Serializes Network configuration data to json."""

    # Get network from data dictionary
    net = simulations_data[simulation_name]["net"]

    # Write to buffer
    with io.StringIO() as output:
        write_network_configuration(net, output)
        return output.getvalue()


def _create_zip(csv_data_list, simulation_name):
    # Zip all files and keep it in memory
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for index, csv_data in enumerate(csv_data_list):
                zf.writestr(f"{simulation_name}_{index + 1}.csv", csv_data)
        zip_buffer.seek(0)
        return zip_buffer.read()