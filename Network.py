from utils.loss_functions import *

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivate = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function):
        #self.loss_function = loss_function
        self.loss = mse
        self.loss_derivate = mse_derivate

    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_derivate(y_train[j], output) 
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def predict(self, input_data):
        pass