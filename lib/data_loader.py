import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist(limit=None):
    """
    Loads MNIST data, normalizes it, and reshapes it for the custom library.
    """
    # Load from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize (0-255 -> 0-1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten (28x28 -> 784)
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Limit dataset size (Optional for speed)
    if limit:
        x_train = x_train[:limit]
        y_train = y_train[:limit]
        x_test = x_test[:limit]
        y_test = y_test[:limit]

    print(f"MNIST Loaded: Train shape {x_train.shape}, Test shape {x_test.shape}")
    return x_train, y_train, x_test, y_test