class Network:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs, learning_rate):
        pass

    def predict(self, input_data):
        pass