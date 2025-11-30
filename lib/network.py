class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        # Forward pass through all layers
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, optimizer):
        for i in range(epochs):
            loss_val = 0
            
            # Forward pass
            output = self.predict(x_train)
            
            # Calculate loss
            loss_val = self.loss(y_train, output)
            
            # Backward pass
            grad = self.loss_prime(y_train, output)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
            
            # Update weights
            optimizer.step(self.layers)
            
            if (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {loss_val:.6f}")