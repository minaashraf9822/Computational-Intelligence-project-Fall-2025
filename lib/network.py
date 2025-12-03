import numpy as np

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
    
    def train(self, x_train, y_train, epochs, optimizer, batch_size=32):
        # Initialize lists to store history
        loss_history = []
        accuracy_history = []

    def train(self, x_train, y_train, epochs, optimizer, batch_size=32):
        loss_history = []
        accuracy_history = []
        
        for i in range(epochs):
            # Shuffle data to prevent cycles
            indices = np.random.permutation(len(x_train))
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Loop through the data in small batches
            for j in range(0, len(x_train), batch_size):
                # Create batch
                x_batch = x_shuffled[j : j + batch_size]
                y_batch = y_shuffled[j : j + batch_size]
                
                # Forward
                output = self.predict(x_batch)
                
                # Backward
                grad = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                
                # Update
                optimizer.step(self.layers)
                
                epoch_loss += self.loss(y_batch, output)
            
            # Average loss for the epoch
            avg_loss = epoch_loss / (len(x_train) / batch_size)
            loss_history.append(avg_loss)
            
            # Simple soft accuracy tracking
            # (Calculated on the last batch of the epoch to save time)
            acc = 1 - np.mean(np.abs(y_batch - output))
            accuracy_history.append(acc)

            if (i + 1) % 1 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {avg_loss:.4f}")
                
        return loss_history, accuracy_history