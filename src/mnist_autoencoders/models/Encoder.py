import torch
import torch.nn as nn
from torchvision.transforms import v2

class Encoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2), # 28->14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), # 14->7
            nn.Flatten(),
            nn.LazyLinear(latent_dims)
        )

    def forward(self, x):
        return self.net(x)
