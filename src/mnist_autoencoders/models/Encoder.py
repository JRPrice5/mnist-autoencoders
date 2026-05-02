import torch
import torch.nn as nn
from torchvision.transforms import v2

class Encoder(nn.Module):
    def __init__(self, cfg, variational=False) -> None:
        super().__init__()
        self.variational = variational
        self.net = nn.Sequential(
            nn.Conv2d(1, cfg.model.conv1.channels, cfg.model.conv1.kernel_size, padding=cfg.model.conv1.padding, stride=cfg.model.conv1.stride), 
            nn.ReLU(),
            nn.Conv2d(cfg.model.conv1.channels, cfg.model.conv2.channels, cfg.model.conv2.kernel_size, padding=cfg.model.conv2.padding, stride=cfg.model.conv2.stride), 
            nn.Flatten(),
            nn.LazyLinear(cfg.model.latent_dim),
        )
        self.std_act = nn.Softplus()
        self.std_layer = nn.LazyLinear(cfg.model.latent_dim)
        self.mean_layer = nn.LazyLinear(cfg.model.latent_dim)

    def forward(self, x):
        hidden = self.net(x)
        if self.variational:
            return self.mean_layer(hidden), self.std_act(self.std_layer(hidden)) + 1e-7
        return hidden
