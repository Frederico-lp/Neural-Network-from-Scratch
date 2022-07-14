import numpy as np

np.random.seed(2)

from Network import Network
from layers.ActivationLayer import ActivationLayer
from layers.Dense import Dense

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

if __name__ == "__main__":
    # nn = Network()
    # nn.add(Dense(2,3))
    # nn.add(ActivationLayer("ReLU"))
    # nn.add(Dense(3,3))
    # nn.add(ActivationLayer("ReLU"))

    # nn.fit()

    # nn.predict()
    X, y = spiral_data(100,3)

    l1 = Dense(2,3)
    l1a = ActivationLayer("ReLU")

    l2 = Dense(3,3)
    l2a = ActivationLayer("ReLU")

    r = l1.forward_propagation(X)
    r = l1a.forward_propagation(r)

    r = l2.forward_propagation(r)
    r = l2a.forward_propagation(r)

    print(r[:5])




