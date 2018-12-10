from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import io
import sys
import random
import numpy as np
from PIL import Image

from . import model
from . import utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

WORKING_DIR = os.getcwd()

def get_args():
    """Argument parser.
  	Returns:
  	  Dictionary of arguments.
  	"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='GCS location to write checkpoints and export models')
    parser.add_argument(
      '--points-path',
      type=str,
      help='Point Annotations Set text file')
    parser.add_argument(
      '--images-input-path',
      type=str,
      help='GCS path to input images')
    parser.add_argument(
      '--num-epochs',
      type=float,
      default=5,
      help='number of times to go through the data, default=5')
    parser.add_argument(
      '--batch-size',
      default=128,
      type=int,
      help='number of records to read during each training step, default=128')
    parser.add_argument(
      '--learning-rate',
      default=.01,
      type=float,
      help='learning rate for gradient descent, default=.01')
    return parser.parse_args()

# Given a label dictionary and a batch size, create and return:
# A label dictionary containing a set of tuples [[imageArr,id],[imageArr,id]...]
# the ORIGINAL DICTIONARY with the chosen images removed
# The max number of images that were added per label,
    #---->this is for if a label has less images than another, so that the user
    #---->will not train more images of one label than another, which would cause weighting issues.
    #---->IE label1: 5 images, label2: 10 images, label3: 7 images.
    #---->5 images of each label would be added to the output dictionary and the number 5 would be returned
def CreateBatchOfImages(batchSize, labelDict, imageInputPath):
    ImageLabelDict = {}
    SuccessNum = 0
    for index in range(batchSize):
        for labelkey in labelDict.keys():
            if labelkey not in ImageLabelDict:
                ImageLabelDict.setdefault(labelkey, [])

            #force add one image, keep trying until one has been added, protects against corrupted images
            while len(labelDict[labelkey]) != 0:
                try:
                    imageId = labelDict[labelkey].pop(0)
                    uriInp = imageInputPath + imageId + ".jpg"

                    with utils._open_file_read_binary(uriInp) as f:
                        image_bytes = f.read()
                        img = Image.open(io.BytesIO(image_bytes))
                        imgArr = np.array(img)

                except: #find image exception error for better style
                    continue
                else:
                    ImageLabelDict[labelkey].append([imgArr, imageId])
                    break
        #make sure all have nonzero number of images left
        for labelkey in labelDict.keys(): #This could be expensive, may want to find better way
            if len(labelDict[labelkey])==0:
                return ImageLabelDict, labelDict, SuccessNum #sucsess num could have offby1error keep an eye out
        SuccessNum += 1

    return ImageLabelDict, labelDict, SuccessNum

# This function returns the data split into training and testing
def prepare_imagedata(ImageLabelDict):
    ###
    ## Extract training and testing data from dict
    ####
    keys = list(ImageLabelDict.keys())
    data = []
    labels = []
    label = 0
    for val in keys:
        imgs = ImageLabelDict.get(val)
        for img in imgs:
            vec = img[0]
            data.append(vec)
            labels.append(label)

        label += 1

    ########
    ##
    ##Split data into training and testing wtih 0.75 train vs 0.25 test
    ########

    total = len(data)
    train_size = round(total*0.75) # must be int type for np.random.choice
    test_size = total-train_size
    random.seed(4)

    train_ind = np.random.choice(len(labels), int(train_size), replace=False)
    test_ind = np.setdiff1d(list(range(0, total)), train_ind)

    labels_test = np.array(labels)
    data_test = np.array(data)

    X_train = data_test[np.where(train_ind)]
    X_test = data_test[np.where(test_ind)]
    Y_train = labels_test[np.where(train_ind)]
    Y_test = labels_test[np.where(test_ind)]

    # We reshape the input data to have a depth of 1 (grey scale)
    X_train = X_train.reshape(X_train.shape[0], 256, 256, 1)
    X_test = X_test.reshape(X_test.shape[0], 256, 256, 1)

    print("X_train shape:" + str(X_train.shape))
    print("X_test shape:" + str(X_test.shape))
    print("Y_train shape:" + str(Y_train.shape))
    print("Y_test shape:" + str(Y_test.shape))

    return (X_train, Y_train), (X_test, Y_test)


def train_and_evaluate(hparams):
    """Helper function: Trains and evaluates model.
    Args:
      hparams: (dict) Command line parameters passed from task.py
    """

    # Perform work to get the ImageLabelDict
    imagePointFile = utils._load_data(path=hparams.points_path, destination='PointAnnotationsSet256x256.txt')
    # Create points from JSON: IDS are key, contains label/confidence/crop
    points = utils.read_file_JSON(filename=imagePointFile)
    # Creates a dictionary such that the key is a label, returns all IDS of that label
    IdsFromLabels = utils.CreateDictLabels(Points=points) 
    # The CreateBatch function after this removes ids from the dictionary, so keep an eye on that
    idsToRemoveFromEachBatch = IdsFromLabels.copy()
    # Creates a batch in someLabelDict in the format (ImageArr, ID)
    someLabelDict, idsToRemoveFromEachBatch, SuccessNum = \
        CreateBatchOfImages(batchSize=50,
                            labelDict=idsToRemoveFromEachBatch,
                            imageInputPath=hparams.images_input_path)


    # Loads data.
    (train_images, train_labels), (test_images, test_labels) = prepare_imagedata(ImageLabelDict=someLabelDict)

    # NOTE: 99% of the code below this is exactly from the tutorial example.
    # https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/mnist/tensorflow/keras/fashion/trainer/task.py

    # Scale values to a range of 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define training steps.
    train_steps = hparams.num_epochs * len(
        train_images) / hparams.batch_size

    # Create TrainSpec.
    train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: model.input_fn(
            train_images,
            train_labels,
            hparams.batch_size,
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=train_steps)

    # Create EvalSpec.
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
    # Shape numpy array.
    test_labels = np.asarray(test_labels).astype('int').reshape((-1, 1))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn(
            test_images,
            test_labels,
            hparams.batch_size,
            mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        exporters=exporter,
        start_delay_secs=10,
        throttle_secs=10)

    # Define running config.
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
    # Create estimator.
    estimator = model.keras_estimator(
      model_dir=hparams.job_dir,
      config=run_config,
      learning_rate=hparams.learning_rate)

    # Start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    """Main function
    """
    args = get_args()

    # LOGGING
    tf.logging.set_verbosity(tf.logging.INFO)

    hparams = hparam.HParams(**args.__dict__)
    train_and_evaluate(hparams)
