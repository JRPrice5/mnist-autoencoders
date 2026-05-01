import torch
import torch.nn as nn
from torchvision.transforms import v2

class Encoder(nn.Module):
    def __init__(self, latent_dims, variational=False) -> None:
        super().__init__()
        self.variational = variational
        self.latent_dims = latent_dims
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2), # 28->14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), # 14->7
            nn.Flatten(),
            nn.LazyLinear(2*latent_dims if variational else latent_dims),
        )
        self.std_act = nn.Softplus()
        self.std_layer = nn.LazyLinear(latent_dims)
        self.mean_layer = nn.LazyLinear(latent_dims)

    def forward(self, x):
        hidden = self.net(x)
        if self.variational:
            return self.mean_layer(hidden), self.std_act(self.std_layer(hidden))
        return hidden
