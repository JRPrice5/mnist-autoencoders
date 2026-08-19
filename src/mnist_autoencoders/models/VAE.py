import torch
import torch.distributions as dist
import torch.nn as nn

from mnist_autoencoders.models.Decoder import Decoder
from mnist_autoencoders.models.Encoder import Encoder


class VAE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg, variational=True)
        self.decoder = Decoder(cfg)
        self.latent_means: torch.Tensor
        self.latent_stds: torch.Tensor
        self.construction_loss: torch.Tensor
        self.kl_loss: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.latent_means, self.latent_stds = self.encoder(x)
        sample = dist.normal.Normal(self.latent_means, self.latent_stds).rsample()
        return self.decoder(sample)
