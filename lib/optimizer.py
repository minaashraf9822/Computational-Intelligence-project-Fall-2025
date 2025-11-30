class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            # Only update if the layer has weights (Dense layers)
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias