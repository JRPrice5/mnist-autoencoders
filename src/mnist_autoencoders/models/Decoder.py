import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.model.latent_dim, cfg.model.conv2.channels*7*7),
            nn.Unflatten(1, (cfg.model.conv2.channels, 7, 7)),
            nn.ConvTranspose2d(cfg.model.conv2.channels, cfg.model.conv1.channels, cfg.model.conv2.kernel_size, stride=cfg.model.conv2.stride, padding=cfg.model.conv2.padding, output_padding=cfg.model.conv2.output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(cfg.model.conv1.channels, 1, cfg.model.conv1.kernel_size, stride=cfg.model.conv1.stride, padding=cfg.model.conv1.padding, output_padding=cfg.model.conv1.output_padding,)
        )

    def forward(self, x):
        return self.net(x)
