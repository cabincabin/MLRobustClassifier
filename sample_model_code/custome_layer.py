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
        
        ######################
        # Generates tensor initialized to a constant value
        # keras.initializers.Constant(value=0) 

        #####################
        # Any function that takes in a weight matrix and returns a loss contribution tensor can be used as a regularizer:
        """
        from keras import backend as K

        def l1_reg(weight_matrix):
        return 0.01 * K.sum(K.abs(weight_matrix))

        model.add(Dense(64, input_dim=64, kernel_regularizer=l1_reg))
        """
        ####################
        # The constraint function constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value.
        # keras.constraints.MaxNorm(max_value=2, axis=0)


        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)

        super(MyLayer, self).build(input_shape)
        

    def call(self, x):
        # Dot product between weights and input 
        return K.dot(x, self.kernel)


    def compute_output_shape():
        return (input_shape[0], self.output_dim)
