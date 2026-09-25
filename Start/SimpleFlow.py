import numpy as np
from sklearn.datasets import fetch_openml
np.random.seed(42)

# Importing the minist handdrawn numbers data set
X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False,
    parser="liac-arff"
)
X = X / 255 # Converting pixeles from 0 -> 255 to 0 -> 1
y = y.astype(int)

# print(X.shape, y.shape)
# print(y[:10], X[:10])


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

class Activation_SoftMax:
    def forward(self, inputs):
        exp_val = np.exp(inputs) 
        predictions = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = predictions

class Loss:
    def calculate(self, predictions, lables):
        correct_probability = predictions[0, lables[0]]
        loss = -np.log(correct_probability)
        self.output = loss

    

# Input Layer
inputLayer = Layer_Dense(784, 784)
inputLayer.forward(X[:1]) #Passing only the first smple
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
OutActivation = Activation_SoftMax()
OutActivation.forward(OutputLayer.output)
# print(OutActivation.output)
loss = Loss()
loss.calculate(OutActivation.output, y)
print(loss.output)
