# Hello-World of Deep Learning (NN = Neural Network)
This project is a hello world to Deep Learning. The project uses the minist 28x28 dataset of handrawn numbers. The Dense Neural Network's job is to make a guess on what the the number is on the input image.

## Activation functions
The project makes use of ReLu activation function on the hidden layers to allow the NN. (Neural Network) to learn in a non-linear way, the consequence of that is a NN that can better adapt pattern in the input images. I also make use of the SoftMax function on the output layer. This is important because all the predictions on the output layer need to equal to 1. SoftMax is also the most common.

## Backpropogatoin
All the backpropogation on this project is done manually. So no autograd engine (done that in another project)

## Optimizer
This project is using **Stochastic Gradient Descent (SGD)** to actualy update the weights and biases that are calculated in the backpropogation process.

## Loss
I decided to go with the **Cross-Entropy Loss function** for this project, as it works best with the softmax prediction on the output layer of the neural network. 

## Final
The result is a DeepLearning model that can find patterns in handrawn numbers and teach it self to predict never seen before hand written numbers.

## Structure
This is how the neural network looks like for this project:


**Training configuration:**
- Framework: N/A (Pure Python)
- Dataset: MNIST
- Training/Test Split: 80% / 20%
- Optimizer: Mini-Batch SGD
- Learning Rate: 0.01
- Batch Size: 50
- Epochs: 20
- Loss Function: Cross-Entropy Loss