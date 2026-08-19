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
from omegaconf import OmegaConf
import wandb
import torch
from PIL import Image

# %%
# Import autoencoder utility tools
from mnist_autoencoders.data.utils import train_loader, test_loader, output_to_image
from mnist_autoencoders.models.VAE import VAE
from mnist_autoencoders.utils import vae_inference_utils

# %%
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

# %%
install(show_locals=True)
cfg = OmegaConf.load("../configs/vae.yaml")
wandb.init(
    project="mnist-autoencoders",
    name=cfg.run_name,
    config=OmegaConf.to_container(cfg, resolve=True),
)
vae = VAE(cfg).to(device)

# %%
# Now i need to define the model and an optimiser
optimiser = torch.optim.Adam(
    vae.parameters(), lr=cfg.training.lr, betas=(0.9, 0.999), eps=1e-8
)

# %%
# Now i need to define the training loop and, inside, the loss function
vae_inference_utils = vae_inference_utils(vae, cfg, device, wandb, optimiser=optimiser)
for epoch in range(cfg.training.epochs):
    vae_inference_utils.train_test_loop(
        mode="train", epoch=epoch, data_loader=train_loader
    )
    with torch.inference_mode():
        vae_inference_utils.train_test_loop(
            mode="test", epoch=epoch, data_loader=test_loader
        )
wandb.finish()

# %% [markdown]
# # Results

# %%
# Here I shall test the model visually, passing an image through,
# denormalising the result and rendering it with pillow
data = next(iter(train_loader))[0][1].to(device)

output = vae(torch.reshape(data, (1, 1, 28, 28)))[0][0]
restored_output = output_to_image(output)
img_hat = Image.fromarray(restored_output.cpu().detach().numpy())

restored_data = output_to_image(data)[0]
img = Image.fromarray(restored_data.cpu().numpy())

# %%
img

# %%
img_hat

# %%
torch.save(vae.state_dict(), "../models/vae_v1/checkpoint_epoch_50.pt")
