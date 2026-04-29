import torch
import torch.nn as nn
from mnist_autoencoders.models.Encoder import Encoder
from mnist_autoencoders.models.Decoder import Decoder

criterion = nn.MSELoss()

class AutoEncoder(nn.Module):
    def __init__(self, latent_dims=None) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x))