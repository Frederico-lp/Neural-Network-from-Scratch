import numpy as np

# Fully Connected Layer (Dense?)
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = 0.10 * np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error