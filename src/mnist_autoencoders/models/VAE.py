import torch
import torch.nn as nn
import torch.distributions as dist
from mnist_autoencoders.models.Encoder import Encoder
from mnist_autoencoders.models.Decoder import Decoder

mse=nn.MSELoss()
kl=nn.KLDivLoss(reduction='batchmean')

class VAE(nn.Module):
    def __init__(self, latent_dims=None) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = Encoder(latent_dims, variational=True)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        means, stds = self.encoder(x)
        self.means = means
        self.stds = stds
        sample = dist.normal.Normal(self.means, self.stds).rsample()
        return self.decoder(sample)
