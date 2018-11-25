import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape
# (60000, 28, 28) this means we have 60,000 samples and all the images are 28 pixels by 28 pixels

plt.imshow(X_train[0])

# We reshape the input data to have a depth of 1 (grey scale)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print X_train.shape
# (60000, 1, 28, 28)

# Then we convert the data type to float

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

 # Then we normalize it so that the values are between 0 and 1
X_train /= 255
X_test /= 255

# Defining the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
# 32 is the  number of convolutional filters to use. Frist 3 is the number of rows in each convolution kernel and second 3 is the number of columns in each kernel.







# Using InceptionV3

#model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

#model.add(Dense(64))
#model.add(Activation('tanh'))

