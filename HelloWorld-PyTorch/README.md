# Hello-World on PyTorch
This is a translation of my Hello World project, the only diffrence is that this one is created with PyTorch. Just as that project this one uses the `mnist` 28x28 dataset of hand drawn images. On which the model learns patterns and learns to recognise new nver seen before handrawn numbers.

## Activation functions
The project makes use of ReLu activation function on the hidden layers to allow the NN. (Neural Network) to learn in a non-linear way, the consequence of that is a NN that can better adapt pattern in the input images. I also make use of the SoftMax function on the output layer (With the corss-entropy loss function). This is important because all the predictions on the output layer need to equal to 1.

## Backpropogatoin
All the backpropogation is handeld by PyTorch's Autograd engine. It works just like my mini autograd engine project just the diffrence being PyTorchs's autoggrad engien is much more smarter and optimized.

## Optimizer
This project is using **Mini-Batch Stochastic Gradient Descent (SGD)** to actualy update the weights and biases that are calculated in the backpropogation process.

## Loss
I decided to go with the **Cross-Entropy Loss function** for this project, as it works best with the softmax prediction on the output layer of the neural network. 

## Final
The result is a DeepLearning model that can find patterns in handrawn numbers and teach it self to predict never seen before hand written numbers.

## Structure
This is how the neural network looks like for this project: