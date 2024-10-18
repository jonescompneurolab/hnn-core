"""Script converting legacy param and json parameter files to hdf5.
This script downloads the legacy files directly from the GitHub repositories.
"""
# Author: George Dang <george_dang@brown.edu.com>

import os
import requests
import shutil
import tempfile
from pathlib import Path
from requests.exceptions import HTTPError

from hnn_core import convert_to_json

root_path = Path(__file__).parents[1]


def download_folder_contents(owner, repo, path):
    """Download files from a GitHub repository folder into temporary space

    Parameters
    ----------
    owner : str
        github account
    repo : str
        repository name
    path : str
        path to directory

    Returns
    -------
    Path to temporary directory or None
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as e:
        raise e

    temp_dir = tempfile.mkdtemp()
    contents = response.json()
    for item in contents:
        if item['type'] == 'file':
            download_url = item['download_url']
            file_name = os.path.join(temp_dir, item['name'])
            with open(file_name, 'wb') as f:
                f.write(requests.get(download_url).content)
                print(f"Downloaded: {file_name}")
    return temp_dir


def convert_param_files_from_repo(owner, repo, repo_path, local_path):
    """Converts param and json parameter files to a hdf5 file.

    Parameters
    ----------
    owner : str
        github account
    repo : str
        repository name
    path : str
        path to directory

    Returns
    -------
    None
    """
    # Download param files
    temp_dir = download_folder_contents(owner, repo, repo_path)
    # Get list of json and param files
    file_list = [Path(temp_dir, f)
                 for f in os.listdir(temp_dir)
                 if f.endswith('.param') or f.endswith('.json')]
    # Assign output location and names
    output_dir = Path(local_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filenames = [Path(output_dir, f.name.split('.')[0])
                        for f in file_list]

    [convert_to_json(file, outfile)
     for (file, outfile) in zip(file_list, output_filenames)]

    # Delete downloads
    shutil.rmtree(temp_dir)


if __name__ == '__main__':

    # hnn param files
    convert_param_files_from_repo(owner='jonescompneurolab',
                                  repo='hnn',
                                  repo_path='param',
                                  local_path=(root_path /
                                              'network_configuration'),
                                  )
    # hnn-core json files
    convert_param_files_from_repo(owner='jonescompneurolab',
                                  repo='hnn-core',
                                  repo_path='hnn_core/param',
                                  local_path=(root_path /
                                              'network_configuration'),
                                  )
