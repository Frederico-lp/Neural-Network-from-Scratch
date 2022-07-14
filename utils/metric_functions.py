import numpy as np
def accuracy(outputs, targets):
    predicions = np.argmax(outputs, axis=1)
    return np.mean(predicions == targets)

