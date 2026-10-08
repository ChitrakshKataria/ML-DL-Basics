# Hello World on PyTorch

This is a PyTorch version of my original Hello World neural network project. Just like the original, this project uses the MNIST dataset, which contains 28×28 pixel images of handwritten digits.

The model learns patterns from these images and uses what it has learned to recognize handwritten digits it has never seen before.

## Activation Functions

The project uses the **ReLU (Rectified Linear Unit)** activation function in the hidden layers. ReLU introduces non-linearity, allowing the neural network to learn more complex patterns in the input images. The output layer produces 10 raw scores, one for each digit (0–9). During training, I use **Cross-Entropy Loss**, which internally applies LogSoftmax. This means I don't need to apply Softmax separately in the model.

## Backpropagation

All backpropagation is handled by PyTorch's **Autograd engine**.

It works on the same fundamental principles as my own mini autograd engine project. The difference is that PyTorch's Autograd is much more optimized and supports more complex operations.

Autograd automatically calculates the gradients of the loss with respect to the model's weights and biases.

## Optimizer

This project uses **Mini-Batch Stochastic Gradient Descent (SGD)** to update the weights and biases of the neural network. The dataset is divided into smaller batches of 50 images. For each batch, the model makes predictions, calculates the loss, and performs backpropagation. The SGD optimizer then uses the calculated gradients and the learning rate to update the model's parameters, gradually reducing the loss.

## Loss

I decided to use the **Cross-Entropy Loss function** because this is a multi-class classification problem with 10 possible classes (digits 0–9). Cross-Entropy measures how well the model's predictions match the correct labels. PyTorch's `nn.CrossEntropyLoss()` combines LogSoftmax and Negative Log-Likelihood Loss into one function.

## Final

The result is a deep learning model trained to recognize handwritten digits. Through training, the neural network learns patterns from labeled images and can use those patterns to predict digits in new, previously unseen images.

## Structure
This is how the neural network looks like for this project:


**Training configuration:**
- Framework: PyTorch
- Dataset: MNIST
- Training/Test Split: 80% / 20%
- Optimizer: Mini-Batch SGD
- Learning Rate: 0.1
- Batch Size: 50
- Epochs: 20
- Loss Function: Cross-Entropy Loss