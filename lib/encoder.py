from lib.network import Network
from lib.layers import Dense
from lib.activations import ReLU

def build_encoder(input_size=784, latent_size=64):
    """
    Creates the Encoder part of the Autoencoder.
    Architecture: Input(784) -> Dense(128) -> ReLU -> Dense(64) -> ReLU
    """
    encoder = Network()
    
    # 1. First Compression Layer (784 -> 128)
    encoder.add(Dense(input_size, 128))
    encoder.add(ReLU())
    
    # 2. Latent Space Layer (128 -> 64)
    encoder.add(Dense(128, latent_size))
    encoder.add(ReLU())
    
    return encoder