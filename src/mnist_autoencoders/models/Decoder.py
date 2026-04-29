import torch
import torch.nn as nn
from torchvision.transforms import v2

class Decoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dims, 32*7*7),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        return self.net(x)
