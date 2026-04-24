import numpy as np
import torchvision
import torch
import torch.nn as nn
from torchvision.transforms import v2

mean=0.1307
std=0.3081
batch_size=64

transform=v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[mean], std=[std])
])

train_data = torchvision.datasets.MNIST('../data', download=True, transform=transform)
test_data = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4)
def output_to_image(x):
    denormalised = std * x + mean
    return (255 * torch.clamp(denormalised, 0, 1)).to(torch.uint8)
    
