import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from mnist_autoencoders.models.Encoder import Encoder
from mnist_autoencoders.models.Decoder import Decoder

mse=nn.MSELoss()
kl=nn.KLDivLoss(reduction='batchmean')

class VAE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg, variational=True)
        self.decoder = Decoder(cfg)

    def forward(self, x):
        latent_means, latent_stds = self.encoder(x)
        self.latent_means = latent_means
        self.latent_stds = latent_stds
        sample = dist.normal.Normal(self.latent_means, self.latent_stds).rsample()
        return self.decoder(sample)

    def calculate_loss(self, x_hat, x_train):
        latent_nrml = torch.stack([self.latent_means, self.latent_stds])
        standard_nrml = torch.zeros(latent_nrml.shape, dtype=torch.float32, device=next(self.parameters()).device)
        standard_nrml[1, :, :] = 1
        latent_log_nrml = F.log_softmax(latent_nrml)
        loss = mse(x_hat, x_train) + self.cfg.training.beta * kl(latent_log_nrml, standard_nrml)
        return loss
