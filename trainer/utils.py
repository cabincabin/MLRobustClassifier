### Shoop code

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import numpy as np
import os
import subprocess

WORKING_DIR = os.getcwd()

# ignore, was part of tutorial
#FASHION_MNIST_TRAIN = 'train-images-idx3-ubyte.gz'
#FASHION_MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
#FASHION_MNIST_TEST = 't10k-images-idx3-ubyte.gz'
#FASHION_MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

# should probably go in task.py
#pointsOutput = "gs://wpiopenimageskaggle/imageIds/PointAnnotationsSet256x256.txt"

def read_file_JSON(filename):
    with open(filename, 'r') as file:
        # self.response.write(cloudstorage_file.read())
        annotString = file.read()
        file.close()

    annotations = json.loads(annotString)
    return annotations

# Clayton function
def getImageGCPPaths(imgTrainingPath, imgTestPath, pointsOutput):
    """
    Should return the training images, training labels, test images, test labels
    """
    points = read_file_JSON(pointsOutput)
    badIndexes = []
    for index, point in enumerate(points):
        uriInp = imageInputPath + point['id'] + ".jpg"
        uricrop = imageOutputScaled + point['id'] + ".jpg"
        uricropEdge = imageOutputEdge + point['id'] + ".jpg"
        #label.append(point['label'])

        size = 256,256
        crop = 0,0,256,256
        if index%100 == 0:
        	print(index)
        try:

            with _open_file_read_binary(uriInp) as f:
                image_bytes = f.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('L')
                img = img.resize(size, Image.ANTIALIAS)
                imgWEdges = img.copy().filter(ImageFilter.FIND_EDGES)
                with file_io.FileIO(uricrop, mode = 'w') as fThumb:
                    img.save(fThumb, "JPEG")
                with file_io.FileIO(uricropEdge, mode = 'w') as fEdge:
                    imgWEdges.save(fEdge, "JPEG")

        except:
            badIndexes.append(index)
    for index in sorted(badIndexes, reverse=True):
        points.pop(index)

    with file_io.FileIO(pointsOutput, mode='w') as pointSaveJson:
        pointSaveJson.write(json.dumps(points))


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
  raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
    for local_file_name in local_file_names
    ]
  for i, gcs_input_path in enumerate(gcs_input_paths):
    if gcs_input_path:
      subprocess.check_call(
        ['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

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
    download_files_from_gcs(path, destination=destination)
    return destination
  return path


def prepare_data(train_file, train_labels_file, test_file, test_labels_file):
  """Loads MNIST Fashion files.
    License:
        The copyright for Fashion-MNIST is held by Zalando SE.
        Fashion-MNIST is licensed under the [MIT license](
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
  Args:
    train_file: (str) Location where training data file is located.
    train_labels_file: (str) Location where training labels file is located.
    test_file: (str) Location where test data file is located.
    test_labels_file: (str) Location where test labels file is located.
  Returns:
    A tuple of training and test data.
  """
  train_labels_file = _load_data(train_labels_file, FASHION_MNIST_TRAIN_LABELS)
  train_file = _load_data(train_file, FASHION_MNIST_TRAIN)
  test_labels_file = _load_data(test_labels_file, FASHION_MNIST_TEST_LABELS)
  test_file = _load_data(test_file, FASHION_MNIST_TEST)

  with gzip.open(train_labels_file, 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(train_file, 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(test_labels_file, 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(test_file, 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)