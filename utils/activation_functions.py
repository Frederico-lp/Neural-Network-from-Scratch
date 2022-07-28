import numpy as np

def ReLU(inputs):
    return np.maximum(0, inputs)

def sigmoid(inputs):
    return 1/(1 + np.exp(-inputs))

def softmax(inputs):
    # prevent overflow with np.max(inputs, axis=1, keepdims=True)
    exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
    
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x);


def ReLU_derivate(inputs):
    return np.heaviside(inputs, 1)


def tanh_derivate(x):
    return 1-np.tanh(x)**2