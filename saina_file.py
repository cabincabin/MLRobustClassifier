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
import custome_layer


np.random.seed(123)  # for reproducibility
# Load pre-shuffled MNIST data into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print X_train.shape
# (60000, 28, 28) this means we have 60,000 samples and all the images are 28 pixels by 28 pixels

plt.figure()
plt.imshow(X_train[0])

# We reshape the input data to have a depth of 1 (grey scale)
if keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)



#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print "X_train.shape:",  X_train.shape
# (60000, 1, 28, 28)
print "X_test.shape:",  X_test.shape

# Then we convert the data type to float

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

 # Then we normalize it so that the values are between 0 and 1
X_train /= 255
X_test /= 255

# Defining the model
model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
# 32 is the number of convolutional filters to use. Frist 3 is the number of rows in each convolution kernel and second 3 is the number of columns in each kernel.

print model.output_shape
# (None, 32, 26, 26)

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
print "test loss:", score[0]
print "test accuracy:",  score[1]
# score[0] gives you the test loss and score[1] gives you the accuracy

tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1, write_graph=True, write_grads=True, batch_size=32, write_images=True)

# We can use a call back to look into the internal state of the model during training
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test), callbacks=[tbCallBack])

