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
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageShow 

# %%
# Import autoencoder utility tools
from mnist_autoencoder_utils import Encoder, Decoder, input_transform, output_transform

# %%
# Initialise networks
encoder = Encoder()
decoder = Decoder()

# %%
# Download mnist data
inference_data = torchvision.datasets.MNIST('data', download=True, transform=input_transform)

# %%
# Gather two mnist images
image_1 = inference_data[0][0] # 1x28x28 
image_2 = inference_data[1][0] # 1x28x28

# %%
# Convert images to b, c, w, h form
image_1 = torch.reshape(image_1, (1, 1, 28, 28))
image_2 = torch.reshape(image_2, (1, 1, 28, 28))

# %%
# Encode both into latent tensors
latent_1, latent_2 = encoder(image_1), encoder(image_2)

# %%
# Interpolate latent tensors

# %%
# Combine into batch tensor

# %%
# Decode into interpolation images (normalised)

# %%
# Convert to 0, 255

# %%
# Plot output in 1, 10 figure

# %%
# Observe random visual interpolation (implying the latent dimensions lack meaning)
