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
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_SoftMax:
    def forward(self, inputs):
        exp_val = np.exp(inputs) 
        predictions = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = predictions
    def backward(self, lables):
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
        


# Initializing Layers (The architecture)
layer1 = Layer_Dense(784, 24)
activation1 = Activation_ReLu()

layer2 = Layer_Dense(24,24)
activation2 = Activation_ReLu()

layer3 = Layer_Dense(24, 24)
activation3 = Activation_ReLu()

OutputLayer = Layer_Dense(24, 10)
SoftMax = Activation_SoftMax()

loss = Loss()



#Training loop (SGD):
epochs = 10
batch_size = 50
learning_rate = 0.01

for epoch in range(epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    X_train = X_train[indices]
    y_train = y_train[indices]

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward Propogation
        layer1.forward(X_batch)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        layer3.forward(activation2.output)
        activation3.forward(layer3.output)

        OutputLayer.forward(activation3.output)
        SoftMax.forward(OutputLayer.output)

        # Loss
        loss.calculate(SoftMax.output, y_batch)

        #Backpropogation
        SoftMax.backward(y_batch)

        OutputLayer.backward(SoftMax.dinputs)

        activation3.backward(OutputLayer.dinputs)
        layer3.backward(activation3.dinputs)

        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)

        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # Updating the weights
        layer1.weights -= learning_rate * layer1.dweights
        layer1.biases -= learning_rate * layer1.dbiases

        layer2.weights -= learning_rate * layer2.dweights
        layer2.biases -= learning_rate * layer2.dbiases

        layer3.weights -= learning_rate * layer3.dweights
        layer3.biases -= learning_rate * layer3.dbiases

        OutputLayer.weights -= learning_rate * OutputLayer.dweights
        OutputLayer.biases -= learning_rate * OutputLayer.dbiases

    # Training performance:
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    OutputLayer.forward(activation3.output)
    SoftMax.forward(OutputLayer.output)

    loss.calculate(SoftMax.output, y_train)

    prediction = np.argmax(SoftMax.output, axis=1)
    accuracy = np.mean(prediction == y_train)

    print(
        f"Epoch: {epoch + 1}/{epoch}"
        f"Loss: {loss.output}"
        f"Accuracy: {accuracy}"
    )