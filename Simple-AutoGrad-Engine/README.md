# Simple Autograd Engine
This is a verry simple autograd engine that can track basci operations like addition, subtraction, multiplication and division to make backpropogation more automated. It is buildt in pure python.

## The Goal
The goal was to lean how a more complex autograd engines works by making a smaller version of them.

## How it works
It works by tracking/remembering how eatch scaler came to be and by what operations, so later on when appling the chain rule to find the partial derivative of the loss function with respect to the scalar becomes a simple and much more automated task.

## Limitations
For now it is only compatable with scalars and dose not suport more complex operations like Matrix-Multiplications, Log, Exponent etc...
