from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.python.keras import models

tf.logging.set_verbosity(tf.logging.INFO)


def keras_estimator(model_dir, config, learning_rate):
    """Creates a Keras Sequential model with layers.
    Args:
      model_dir: (str) file path where training files will be written.
      config: (tf.estimator.RunConfig) Configuration options to save model.
      learning_rate: (int) Learning rate.
    Returns:
      A keras.Model
    """

    img_len=256
    img_width=256

    # Defining the model

    model = models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(img_len, img_width, 1)))
    # 32 is the number of convolutional filters to use. First 3 is the number of rows in each convolution kernel and
    # second 3 is the number of columns in each kernel.

    model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter
    # across the previous layer and taking the max of the 4 values in the 2x2 filter.

    model.add(Dropout(0.25))
    # Dropout regularizes the model and prevents overfitting

    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=tf.nn.softmax))


    # Compile model with learning parameters.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator


def input_fn(features, labels, batch_size, mode):
    """Input function.
    Args:
      features: (numpy.array) Training or eval data.
      labels: (numpy.array) Labels for training or eval data.
      batch_size: (int)
      mode: tf.estimator.ModeKeys mode
    Returns:
      A tf.estimator.
    """
    # Default settings for training.
    if labels is None:
      inputs = features
    else:
      # Change numpy array shape.
      inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
      dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def serving_input_fn():
    """Defines the features to be passed to the model during inference.
    Expects already tokenized and padded representation of sentences
    Returns:
      A tf.estimator.export.ServingInputReceiver
    """
    feature_placeholder = tf.placeholder(tf.float32, [None, 256, 256, 1])
    features = feature_placeholder
    return tf.estimator.export.TensorServingInputReceiver(features,
                                                          feature_placeholder)
