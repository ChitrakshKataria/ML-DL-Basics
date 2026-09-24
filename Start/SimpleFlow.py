import numpy as np

X = np.random.randn(1,784)
np.random.seed(42)

class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
    def forward(self, inputs):
        '''
        Function to forward propgate though the NN.
        '''
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Input Layer
inputLayer = Layer_Dense(784, 784)
inputLayer.forward(X)
activatoin1 = Activation_ReLu()
activatoin1.forward(inputLayer.output)

#Hiden Layer 1
HLayer1 = Layer_Dense(784, 16)
HLayer1.forward(activatoin1.output)
Hactivatoin1 = Activation_ReLu()
Hactivatoin1.forward(HLayer1.output)

#Hiden Layer 2
HLayer2 = Layer_Dense(16, 16)
HLayer2.forward(Hactivatoin1.output)
Hactivatoin2 = Activation_ReLu()
Hactivatoin2.forward(HLayer2.output)

#Output Layer
OutputLayer = Layer_Dense(16, 10)
OutputLayer.forward(Hactivatoin2.output)
print(OutputLayer.output)

