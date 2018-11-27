from keras import backend as K
from keras.engine.topology import Layer

class CustomeLayer:

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomeLayer, self).__init__(**kwargs)



    # We define the weights in the build function
    def build(self, input_shape):

        # First we have to create a trainable weight variable for this layer.
        # If input is of the shape (batch_size, input_dim), weights 
        # need to be of the shape (input_dim, output_dim). 
        # In this function output dimension denotes the number of units in the layer.
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)

        super(MyLayer, self).build(input_shape)
        

    def call(self, x):
        return K.dot(x, self.kernel)


    def compute_output_shape():
        return (input_shape[0], self.output_dim)
