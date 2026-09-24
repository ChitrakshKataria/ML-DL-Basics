import numpy as np

X = np.random.randn(1,784)
print(X)

class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Input Layer
inputLayer = Layer_Dense(784, 784)
inputLayer.forward(X)
activatoin1 = Activation_ReLu()
activatoin1.forward(inputLayer.output)
