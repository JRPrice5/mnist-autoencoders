# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: CVAE (3.12)
#     language: python
#     name: cvae
# ---

# %%
import torchvision
import numpy as np
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageShow 

# %% [markdown]
# ## Gather two mnist images

# %%
mean=0.1307
std=0.3081

# %%
transform=v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[mean], std=[std])
])

# %%
# Download latest version
inference_data = torchvision.datasets.MNIST('data', download=True, transform=transform)
comparison_data = torchvision.datasets.MNIST('data', train=False, download=True)

# %%
#(2)import the AE
from mnist_autoencoder import AutoEncoder

# %%
#(3)interpolate the two images

# %%
#(4)encode the two images

# %%
#(5)interpolate the encodings

# %%
#(6)plot both interpolations in a 2, 10 figure
