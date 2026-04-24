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
from MNIST.data.parser import transform, output_to_image
from MNIST.models.ae import AutoEncoder, latent_dims

# %%
interpolation_length = 10

# %%
# Initialise networks
autoencoder = AutoEncoder()
autoencoder.load_state_dict(torch.load('../models/autoencoder_v1/checkpoint_epoch_20.pt', map_location='cpu'))
encoder = autoencoder.encoder
decoder = autoencoder.decoder

# %%
# Download mnist data
inference_data = torchvision.datasets.MNIST('../data', download=True, transform=transform)

# %%
# Gather two mnist images
image_1 = inference_data[0][0] # 1x28x28 
image_2 = inference_data[1][0] # 1x28x28
plt.imshow(image_1.permute(1, 2, 0), cmap='gray')

# %%
# Convert images to b, c, w, h form
image_1 = torch.reshape(image_1, (1, 1, 28, 28))
image_2 = torch.reshape(image_2, (1, 1, 28, 28))

# %%
# Encode both into latent tensors
latent_1, latent_2 = encoder(image_1), encoder(image_2)

# %%
# Interpolate latent tensors
batch = np.empty((interpolation_length, latent_dims))
with torch.inference_mode():
    for i in range(interpolation_length):
        batch[i] = latent_1 + (i*(latent_2-latent_1))/interpolation_length-1
batch

# %%
# Decode into interpolation images (normalised)
formatted_interpolation = torch.from_numpy(batch)
output = decoder(formatted_interpolation.to(torch.float32))
output.shape

# %%
# Convert to 0, 255
images = output_to_image(output)

# %%
# Plot output in 1, 10 figure
fig, axes = plt.subplots(1, 10, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(images[i].permute(1, 2, 0), cmap='gray')
    ax.axis('off')

# %% [markdown]
# ## Observations
# In the above interpolation I can see the transition from a 5 to a 0. Honestly not much looks wrong with it... Either I'm missing a key detail or MNIST is not a good example for showing the lack of meaning within latent dimensions. 

# %% [markdown]
# After a brief AI chat, I have reached the conclusion that there is no continuity between these two encodings. This means that if sampling was attempted, most points would be gibberish and not interpretable digits.
