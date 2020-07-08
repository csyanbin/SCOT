r"""Functions to download semantic correspondence datasets"""

import tarfile
import os

import requests

from . import pfpascal
from . import pfwillow
from . import spair


def load_dataset(benchmark, datapath, thres, device, split='test', cam=''):
    r"""Instantiate desired correspondence dataset"""
    correspondence_benchmark = {
        'pfpascal': pfpascal.PFPascalDataset,
        'pfwillow': pfwillow.PFWillowDataset,
        'spair': spair.SPairDataset,
    }

    dataset = correspondence_benchmark.get(benchmark)
    if dataset is None:
        raise Exception('Invalid benchmark dataset %s.' % benchmark)

    return dataset(benchmark, datapath, thres, device, split, cam)


def download_from_google(token_id, filename):
    r"""Download desired filename from Google drive"""
    print('Downloading %s ...' % os.path.basename(filename))

    url = 'https://docs.google.com/uc?export=download'
    destination = filename + '.tar.gz'
    session = requests.Session()

    response = session.get(url, params={'id': token_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': token_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)
    file = tarfile.open(destination, 'r:gz')

    print("Extracting %s ..." % destination)
    file.extractall(filename)
    file.close()

    os.remove(destination)
    os.rename(filename, filename + '_tmp')
    os.rename(os.path.join(filename + '_tmp', os.path.basename(filename)), filename)
    os.rmdir(filename+'_tmp')


def get_confirm_token(response):
    r"""Retrieve confirm token"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    r"""Save the response to the destination"""
    chunk_size = 32768

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)


def download_dataset(datapath, benchmark):
    r"""Download desired semantic correspondence benchmark dataset from Google drive"""
    if not os.path.isdir(datapath):
        os.mkdir(datapath)

    file_data = {
        'pfwillow': ('1Moes_nP-1AGxuG6TkuEl100NaHGWhRfv', 'PF-WILLOW'),
        'pfpascal': ('1r54YXfCWGHe523w3ZQnCP0Y1TDToKx8n', 'PF-PASCAL'),
        'spair': ('17gXi6iSK6jIz4SGJzrO6ELJGYRAKUkhR', 'SPair-71k')
    }

    file_id, filename = file_data[benchmark]
    abs_filepath = os.path.join(datapath, filename)

    if not os.path.isdir(abs_filepath):
        download_from_google(file_id, abs_filepath)
