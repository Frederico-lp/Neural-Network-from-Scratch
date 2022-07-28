from utils.activation_functions import *

# inherit from base class Layer
class ActivationLayer:
    def __init__(self, activation):
        if(activation == "ReLU"):
            self.activation_fun = ReLU
            self.activation_fun_derivate = ReLU_derivate
        #for now other functions are not implemented
        else:
            self.activation_fun = ReLU
            self.activation_fun_derivate = ReLU_derivate

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_fun(input_data)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_fun_derivate(self.input) * output_error