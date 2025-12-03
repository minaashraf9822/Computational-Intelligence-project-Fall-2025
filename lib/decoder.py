from lib.network import Network
from lib.layers import Dense
from lib.activations import ReLU, Sigmoid

def build_decoder(latent_size=64, output_size=784):
    """
    Creates the Decoder part of the Autoencoder.
    Architecture: Latent(64) -> Dense(128) -> ReLU -> Dense(784) -> Sigmoid
    """
    decoder = Network()
    
    # 1. First Expansion Layer (64 -> 128)
    decoder.add(Dense(latent_size, 128))
    decoder.add(ReLU())
    
    # 2. Output Reconstruction Layer (128 -> 784)
    decoder.add(Dense(128, output_size))
    
    # We use Sigmoid at the end because image pixel values are between 0 and 1
    decoder.add(Sigmoid())
    
    return decoder