import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

np.random.seed(2)

from Network import Network
from layers.ActivationLayer import ActivationLayer
from layers.Dense import Dense


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)


    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    nn = Network()
    nn.add(Dense(28*28, 100))
    nn.add(ActivationLayer("ReLU"))
    nn.add(Dense(100,50))
    nn.add(ActivationLayer("ReLU"))
    nn.add(Dense(50, 10)) 
    nn.add(ActivationLayer("tanh"))

    #for now only one loss is available
    nn.compile("mse")

    nn.fit(x_train[0:2000], y_train[0:2000], epochs=20, learning_rate=0.1)

    pred = nn.predict(x_test[0:10])
    print("predicted values : ")
    print(np.argmax(pred, axis=2))
    print("true values : ")
    print(np.argmax(y_test[0:10], axis=1, keepdims=True))







