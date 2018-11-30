import tensorflow as tf
import json
import io
import os
import subprocess
from tensorflow.python.lib.io import file_io
from PIL import Image
import numpy as np
import argparse
from PIL import ImageFilter
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import matplotlib.image as mpimg



def main(ImageLabelDict, idsToRemoveFromEachBatch, SuccessNum):
    ###
    ## Extract training and testing data from dict
    ####
    keys=list(ImageLabelDict.keys())
    data=[]
    labels=[]
    label=0
    for val in keys:
        imgs=ImageLabelDict.get(val)
        for img in imgs:
            print(img)
            vec=img[0]
            print(vec)
            data.append(vec)
            labels.append(label)
        label+=1
        
    ########
    ##
    ##Split data intro training and testing wtih 0.75 train vs 0.25 test
    ########
        
    total=len(data)
    train_size=round(total*0.75)
    test_size=total-train_size
    import random
    random.seed(4)
  
    train_ind=np.random.choice(len(labels),train_size,replace=False)
    test_ind=np.setdiff1d(list(range(0, total)),train_ind)


    labels_test=np.array(labels)
    data_test=np.array(data)
    
    X_train=data_test[np.where(train_ind)]
    X_test=data_test[np.where(test_ind)]
    Y_train=labels_test[np.where(train_ind)]
    Y_test=labels_test[np.where(test_ind)]
    
    ############
    ###  Specify image size and format data to input into model
    ###
    ############

    img_len=256
    img_width=256
    # We reshape the input data to have a depth of 1 (grey scale)
    if keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_len, img_width)
        X_test = X_test.reshape(X_test.shape[0], 1, img_len, img_width)
        input_shape = (1, img_len, img_width)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_len, img_width, 1)
        X_test = X_test.reshape(X_test.shape[0], img_len, img_width, 1)
        input_shape = (img_len, img_width, 1)


    # Then we convert the data type to float

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Then we normalize it so that the values are between 0 and 1
    X_train /= 255
    X_test /= 255
    
        
    ############
    ###  Run Keras Models
    ###
    ############

    # Defining the model
    model = Sequential()
    model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(img_len, img_width, 1)))
    # 32 is the number of convolutional filters to use. Frist 3 is the number of rows in each convolution kernel and second 3 is the number of columns in each kernel.


    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter.

    model.add(Dropout(0.25))
    # Dropout regularizes the model and prevents overfitting

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # Final layer has the output size of 10 to correspond to the number of classes

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit model to training data
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    # Evaluate the model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("test loss:", score[0])
    print("test accuracy:",  score[1])
    # score[0] gives you the test loss and score[1] gives you the accuracy

    tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1, write_graph=True, write_grads=True, batch_size=32, write_images=True)

    # We can use a call back to look into the internal state of the model during training
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test), callbacks=[tbCallBack])



