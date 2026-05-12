from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MSELoss, KLDivLoss


class vae_inference_utils:
    def __init__(self, model, cfg, device, logger, optimiser):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.optimiser = optimiser
        self.mse = MSELoss()
        self.kl = KLDivLoss(reduction="batchmean")

    def train_test_loop(self, mode="train", epoch=-1, data_loader=None):
        if mode != "train" and mode != "test":
            raise ValueError("mode must be train or test")
        if not data_loader:
            raise ValueError("A pytorch data loader must be provided")

        leave = True if mode == "train" else False
        pbar = tqdm(data_loader, desc=f"epoch: {epoch}", leave=leave)
        losses: dict[str, float] = {"total": 0, "construction": 0, "kl": 0}
        for x, _ in pbar:
            if mode == "train":
                self.optimiser.zero_grad()
            x_hat = self.model(x.to(self.device))
            construction_loss, kl_loss = self._calculate_loss(x_hat, x.to(self.device))
            total_loss = construction_loss + self.cfg.training.beta * kl_loss
            losses["total"] += total_loss.item()
            losses["construction"] += construction_loss.item()
            losses["kl"] += kl_loss.item()
            if mode == "train":
                total_loss.backward()
                self.optimiser.step()
        self.logger.log(
            {
                f"{mode}/total_loss": losses["total"] / len(data_loader),
                f"{mode}/construction_loss": losses["construction"] / len(data_loader),
                f"{mode}/kl_loss": losses["kl"] / len(data_loader),
            },
            step=epoch,
        )

    def _calculate_loss(self, x_hat: Tensor, x_train: Tensor) -> tuple[Tensor, Tensor]:
        latent_nrml = torch.stack([self.model.latent_means, self.model.latent_stds])
        latent_log_nrml = F.log_softmax(latent_nrml)

        standard_nrml = torch.zeros(
            latent_nrml.shape,
            dtype=torch.float32,
            device=self.device,
        )
        standard_nrml[1, :, :] = 1

        construction_loss = self.mse(x_hat, x_train)
        kl_loss = self.kl(latent_log_nrml, standard_nrml)

        return construction_loss, kl_loss
