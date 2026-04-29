import torch
import torch.nn as nn
import torch.distributions.normal
from mnist_autoencoders.models.Encoder import Encoder
from mnist_autoencoders.models.Decoder import Decoder

mse=nn.MSELoss()
kl=nn.KLDivLoss()

class VAE(nn.Module):
    def __init__(self, latent_dims=None) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        means, stds = self.encoder(x)
        sample = Normal(means, stds).rsample()
        return self.decoder(sample)
