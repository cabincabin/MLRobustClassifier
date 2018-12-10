from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import subprocess
import random
import json

from tensorflow.python.lib.io import file_io
from PIL import Image

WORKING_DIR = os.getcwd()

# The following is helper functions the team created
####################################################################

# Use TRUE to deploy to GCP ML engine
# def USEGCP(UseGCP):
#     if UseGCP == True:
#         args = parser.parse_args()
#         subprocess.call(["gsutil", "cp", args.points_path, "PointAnnotationsSet256x256.txt"])
#         global imageInputPath, imagePoint
#         imagePoint = "PointAnnotationsSet256x256.txt"
#         imageInputPath = "gs://wpiopenimageskaggle/Imagefiles256x256/"

# Dumps a file to JSON format, used to get points from 'PointAnnotationsSet.txt'
def read_file_JSON(filename):
    with open(filename, 'r') as file:
        # self.response.write(cloudstorage_file.read())
        annotString = file.read()
        file.close()

    annotations = json.loads(annotString)
    return annotations

# Opens the file url in binary format
def _open_file_read_binary(uri):
    try:
        return file_io.FileIO(uri, mode='r')
    except:
        return file_io.FileIO(uri, mode='rb')

# Given a set of JSON, in form
# [{"id": "9eb39d618fd92994", "annotations": [{"confidence": "1", "label": "/m/01g317", "y1": "0.807014", "y0": "0.218368", "x0": "0.072245", "x1": "0.729939", "id": "9eb39d618fd92994"}]}
# Create a dictionary: labels are the key, returns a list of all of the ids in that label
def CreateDictLabels(Points):
    dictOfLabels = {}
    for point in Points:
        labelkey = point['annotations'][0]['label'] #get the label
        if labelkey not in dictOfLabels:
            dictOfLabels.setdefault(labelkey, [])
        dictOfLabels[labelkey].append(point['id'])
    return dictOfLabels


# The following are helper functions from the tutorial files
####################################################################

def download_files_from_gcs(source, destination):
    """Download files from GCS to a WORKING_DIR/.
    Args:
    source: GCS path to the training data
    destination: GCS path to the validation data.
    Returns:
    A list to the local data paths where the data is downloaded.
    """
    local_file_names = [destination]
    gcs_input_paths = [source]

    # Copy raw files from GCS into local path.
    raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name) for local_file_name in local_file_names]
    for i, gcs_input_path in enumerate(gcs_input_paths):
        if gcs_input_path:
            subprocess.check_call(['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

    return raw_local_files_data_paths

def _load_data(path, destination):    
    """Verifies if file is in Google Cloud.
    Args:
    path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
    destination: (str) The filename to save as on local disk.
    Returns:
    A filename
    """
    if path.startswith('gs://'):
        print("path starts with gs")
        download_files_from_gcs(path, destination=destination)
        return destination

    return path

