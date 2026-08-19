import torch.nn as nn
from mnist_autoencoders.models.Encoder import Encoder
from mnist_autoencoders.models.Decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x):
        return self.decoder(self.encoder(x))
