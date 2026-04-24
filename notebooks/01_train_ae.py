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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageShow 

# %%
# Import autoencoder utility tools
from mnist_autoencoders.data.parser import train_loader, test_loader, output_to_image
from mnist_autoencoders.models.ae import AutoEncoder

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
n_epochs = 50 
lr = 0.001
momentum = 0.9

# %%
train_means = np.empty((n_epochs))
test_means = np.empty((n_epochs))

# %%
train_batches = len(train_loader)
test_batches = len(test_loader)

# %%
# Now i need to define the model and an optimiser
autoencoder = AutoEncoder().to(device)
optimiser = torch.optim.SGD(autoencoder.parameters(), lr=lr, momentum=momentum)
criterion = nn.MSELoss()

# %%
# Now i need to define the training loop and, inside, the loss function
for epoch in range(n_epochs):
    train_loss = 0
    test_loss = 0
    for i, (x_train, _) in enumerate(train_loader):
        optimiser.zero_grad()
        x_hat = autoencoder(x_train.to(device))
        loss = criterion(x_hat, x_train.to(device))
        train_loss += loss.item()
        loss.backward()
        optimiser.step()
    train_means[epoch] = train_loss / train_batches
    with torch.inference_mode():
        for j, (x_test, _) in enumerate(test_loader):
            x_hat = autoencoder(x_test.to(device))
            loss = criterion(x_hat, x_test.to(device))
            test_loss += loss.item()
        test_means[epoch] = test_loss / test_batches
    print(f"\r{f'{epoch+1}/{n_epochs}: {i+1}/{train_batches}':<20}", end='', flush=True)

# %% [markdown]
# # Results

# %%
# Here I shall test the model visually, passing an image through,
# denormalising the result and rendering it with pillow
data = next(iter(train_loader))[0][0].to(device)

output = autoencoder(torch.reshape(data, (1,1,28,28)))[0][0]
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
ax.set(xlabel='Batch', ylabel='MSE Loss')
ax.grid()

# %%
torch.save(autoencoder.state_dict(), '../models/autoencoder_128/checkpoint_epoch_50.pt')
