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
from rich.traceback import install
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageShow 

# %%
# Import autoencoder utility tools
from mnist_autoencoders.data.utils import train_loader, test_loader, output_to_image
from mnist_autoencoders.models.VAE import VAE

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
install(show_locals=True)
cfg = OmegaConf.load('../configs/vae.yaml')
wandb.init(project="mnist-autoencoders", name=cfg.run_name,
           config=OmegaConf.to_container(cfg, resolve=True))
vae = VAE(cfg).to(device)

# %%
# Now i need to define the model and an optimiser
optimiser = torch.optim.Adam(vae.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum)

# %%
# Now i need to define the training loop and, inside, the loss function
for epoch in range(cfg.training.epochs):
    pbar = tqdm(train_loader, desc=f'epoch: {epoch}')
    losses = 0
    for x_train, _ in pbar:
        optimiser.zero_grad()
        x = x_train.to(device)
        x_hat = vae(x)
        loss = vae.calculate_loss(x_hat, x)
        loss.backward()
        losses += loss.item()
        optimiser.step()
    wandb.log({"train/loss": losses/len(train_loader)}, step=epoch)
    with torch.inference_mode():
        testpbar = tqdm(test_loader, leave=False)
        losses = 0
        for x_test, _ in testpbar:
            x = x_test.to(device)
            x_hat = vae(x)
            loss = vae.calculate_loss(x_hat, x)
            losses += loss.item()
        wandb.log({"test/loss": losses/len(test_loader)}, step=epoch)
wandb.finish()

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

# %%
img

# %%
img_hat

# %%
# torch.save(vae.state_dict(), '../models/VAE_128/checkpoint_epoch_10.pt')
