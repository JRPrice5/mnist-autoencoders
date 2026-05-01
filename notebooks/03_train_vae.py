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
#     display_name: mnist-autoencoders
#     language: python
#     name: mnist-autoencoders
# ---

# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageShow 

# %%
# Import autoencoder utility tools
from mnist_autoencoders.data.utils import train_loader, test_loader, output_to_image
from mnist_autoencoders.models.VAE import VAE, mse, kl

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
n_epochs = 10 
lr = 0.001
momentum = 0.9
latent_dims = 128
beta = 0.1

# %%
train_means = np.empty((n_epochs))
test_means = np.empty((n_epochs))

# %%
train_batches = len(train_loader)
test_batches = len(test_loader)

# %%
# Now i need to define the model and an optimiser
vae = VAE(latent_dims).to(device)
optimiser = torch.optim.SGD(vae.parameters(), lr=lr, momentum=momentum)

# %%
# Test to ensure correct std nrml creation
standard_nrml = torch.zeros((2, latent_dims), dtype=torch.float32, device=device)
standard_nrml[0, :].apply_(lambda x: 1)
standard_nrml


# %%
def calculate_vae_loss(x_hat, x_train, vae):
        standard_nrml = torch.zeros((2, latent_dims), dtype=torch.float32, device=device)
        standard_nrml = standard_nrml[0, :].apply_(lambda x: 1)
        # Convert bsxld, bsxld to bsx2xld
        # Add dim 1 to both
        means = torch.unsqueeze(vae.means, dim=1)
        stds = torch.unsqueeze(vae.stds, dim=1)
        # Concatenate along new dimension
        z_dist = torch.cat((means, stds), dim=1)
        input = F.log_softmax(z_dist, dim=1)
        loss = mse(x_hat, x_train) + beta * kl(input, standard_nrml)
        return loss


# %%
# Now i need to define the training loop and, inside, the loss function
torch.autograd.set_detect_anomaly(True, check_nan=False)
for epoch in range(n_epochs):
    train_loss = 0
    test_loss = 0
    for i, (x_train, _) in enumerate(train_loader):
        optimiser.zero_grad()
        x_hat = vae(x_train.to(device))
        loss = calculate_vae_loss(x_hat, x_train.to(device), vae)
        loss.backward()
        train_loss += loss.item()
        optimiser.step()
    train_means[epoch] = train_loss / train_batches
    with torch.inference_mode():
        for j, (x_test, _) in enumerate(test_loader):
            x_hat = vae(x_test.to(device))
            test_loss += calculate_vae_loss(x_hat, x_test.to(device), vae)
        test_means[epoch] = test_loss / test_batches
    print(f"\r{f'{epoch+1}/{n_epochs}: {i+1}/{train_batches}':<20}", end='', flush=True)

# %% [markdown]
# # Results

# %%
# Here I shall test the model visually, passing an image through,
# denormalising the result and rendering it with pillow
data = next(iter(train_loader))[0][1].to(device)

output = vae(torch.reshape(data, (1,1,28,28)))[0][0]
restored_output = output_to_image(output)
img_hat = Image.fromarray(restored_output.cpu().detach().numpy())

restored_data = output_to_image(data)[0]
img = Image.fromarray(restored_data.cpu().numpy())
ImageShow.show(img_hat)
ImageShow.show(img)

# %%
# Once the model roughly works I need to plot loss changes and determine over or under
# fitting
fig, ax = plt.subplots(figsize=[30, 20])
batch_counts = np.arange(train_batches, train_batches*n_epochs + 1, train_batches)
train = ax.plot(batch_counts, train_means.ravel(), color='red', label='Train Loss')
test = ax.plot(batch_counts, test_means.ravel(), color='blue', label='Val Loss')
ax.legend()
ax.set(xlabel='Batch', ylabel='Loss')
ax.grid()

# %%
# torch.save(vae.state_dict(), '../models/VAE_128/checkpoint_epoch_10.pt')
