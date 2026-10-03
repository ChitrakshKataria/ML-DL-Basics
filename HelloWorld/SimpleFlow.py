import numpy as np
from sklearn.datasets import fetch_openml
np.random.seed(42)

# Importing the minist handdrawn numbers data set
X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False,
)
X = X / 255 # Converting pixeles from 0 -> 255 to 0 -> 1
y = y.astype(int)


# Training and Testing Split 80:20 :
split_index = int(len(X) * 0.8)

X_train = X[:split_index] # 80% of the training data
X_test = X[split_index:] # Last 20% of the training data

# Same concept here
y_train = y[:split_index]
y_test = y[split_index:]


class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
    def forward(self, inputs):
        '''
        Function to forward propgate though the NN.
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(dvalues, self.inputs)
        self.dbiases = np.sum(dvalues, axis=1, keepdims=True)
        self.dinputs = np.dot(self.weights.T, dvalues)

class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dinputs(self.inputs <= 0) = 0


class Activation_SoftMax:
    def forward(self, inputs):
        exp_val = np.exp(inputs) 
        predictions = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = predictions
    def backward(self, dvalues, lables):
        sample = len(self.output)
        self.dinputs = self.output.copy()
        self.dinputs[np.arange(sample), lables] -= 1
        self.dinputs /= sample

class Loss:
    def calculate(self, predictions, lables):
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7) # There was a problem with taking the log(0) which is equal to infinity so we just dont allow the predictions to go to 0 we clip them before that
        samples = len(predictions)
        correct_probability = predictions[np.arange(samples), lables[:samples]]
        loss = -np.log(correct_probability)
        self.output = np.mean(loss)
class Accuracy:
    def calculate(self, predictions, lables):
        max_prediction = np.argmax(predictions, axis=1)
        self.output = np.mean(max_prediction == lables)
        

    

# Input Layer
inputLayer = Layer_Dense(784, 784)
inputLayer.forward(X_train) #Passing only the first 20 samples
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
loss.calculate(OutActivation.output, y_train)
# print(loss.output)
acc = Accuracy()
acc.calculate(OutActivation.output, y_train)
print(acc.output)
