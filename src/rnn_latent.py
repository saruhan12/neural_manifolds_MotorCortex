from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class LatentRNN(nn.Module):
    """RNN for modeling dynamics in PCA latent space.

    Simple recurrent architecture: latent input → tanh RNN → linear readout → latent output.
    Trained to predict one-step-ahead latent states, capturing temporal structure
    of neural population dynamics.

    Parameters
    ----------
    d_latent : int
        Dimensionality of latent space (e.g., number of PCs)
    hidden : int, optional
        Hidden state size of RNN (default: 64)
    """

    def __init__(self, d_latent: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=d_latent,
            hidden_size=hidden,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.readout = nn.Linear(hidden, d_latent)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(z)
        return self.readout(out)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step_rnn(
    Z: np.ndarray,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    batch_size: int | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[LatentRNN, Dict[str, np.ndarray]]:
    """
    Train a one-step predictor: z_t -> z_{t+1} with teacher forcing.
    Z shape: (trials, time, d_latent)
    """
    _set_seed(seed)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 3:
        raise ValueError("Z must be 3D: (trials, time, d_latent)")

    trials, time, d_latent = Z.shape
    if time < 2:
        raise ValueError("Z must have time dimension >= 2")

    model = LatentRNN(d_latent=d_latent, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    if batch_size is None or batch_size <= 0:
        batch_size = trials

    train_losses = []
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        perm = rng.permutation(trials)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, trials, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch = Zt[batch_idx]
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :]

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        train_losses.append(epoch_loss / max(1, steps))

    train_log = {"loss": np.asarray(train_losses, dtype=float)}
    return model, train_log


def rollout(model: LatentRNN, z0: np.ndarray, T: int, device: str = "cpu") -> np.ndarray:
    """
    Roll out the latent dynamics from z0 for T steps.
    z0 shape: (batch, d_latent)
    returns: (batch, T, d_latent)
    """
    model.eval()
    with torch.no_grad():
        z = torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(1)
        outputs = []
        h = None
        for _ in range(T):
            out, h = model.rnn(z, h)
            pred = model.readout(out[:, -1, :])
            outputs.append(pred)
            z = pred.unsqueeze(1)
        Zhat = torch.stack(outputs, dim=1)
    return Zhat.cpu().numpy()
