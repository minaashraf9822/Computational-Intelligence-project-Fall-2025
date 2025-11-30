import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        # Returns output
        raise NotImplementedError

    def backward(self, output_gradient):
        # Updates parameters and returns input gradient
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # He initialization or random small numbers
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        
        # Gradients (to be used by the optimizer)
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input_data):
        self.input = input_data
        # Y = XW + b
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient):
        # Calculate gradients for weights and biases
        # dL/dW = X.T * dL/dY
        self.grad_weights = np.dot(self.input.T, output_gradient)
        
        # dL/db = sum(dL/dY)
        self.grad_bias = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Calculate gradient for input to pass to the previous layer
        # dL/dX = dL/dY * W.T
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient