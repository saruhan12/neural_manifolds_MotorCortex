from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.pca import fit_pca
from src.rnn_latent import rollout, train_one_step_rnn


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_pca_or_fit(pca_path: str, preproc_path: str, n_components: int):
    if os.path.exists(pca_path):
        data = np.load(pca_path)
        return data
    pre = np.load(preproc_path)
    rates = pre["rates"]
    pca_res = fit_pca(rates, n_components=n_components)
    os.makedirs(os.path.dirname(pca_path), exist_ok=True)
    np.savez_compressed(
        pca_path,
        components=pca_res.components,
        explained_variance=pca_res.explained_variance,
        explained_variance_ratio=pca_res.explained_variance_ratio,
        scores=pca_res.scores,
        mean=pca_res.mean,
    )
    return np.load(pca_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one-step latent RNN on PCA trajectories.")
    parser.add_argument("--pca-input", default="data/processed/pca.npz")
    parser.add_argument("--preproc-input", default="data/processed/preprocessed.npz")
    parser.add_argument("--d-latent", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot-dir", default="report/figs")
    args = parser.parse_args()

    _set_seed(args.seed)
    pca_data = _load_pca_or_fit(args.pca_input, args.preproc_input, n_components=max(10, args.d_latent))
    Z = pca_data["scores"][:, :, : args.d_latent]

    model, train_log = train_one_step_rnn(
        Z,
        hidden=args.hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size if args.batch_size > 0 else None,
        device=args.device,
        seed=args.seed,
    )

    os.makedirs("data/processed", exist_ok=True)
    torch.save(model.state_dict(), "data/processed/rnn_latent.pt")

    with torch.no_grad():
        Z_pred_teacher = model(torch.tensor(Z[:, :-1, :], dtype=torch.float32, device=args.device))
    Z_pred_teacher = Z_pred_teacher.cpu().numpy()

    Z_rollout = rollout(model, Z[:, 0, :], T=Z.shape[1] - 1, device=args.device)

    bin_size = np.nan
    if os.path.exists(args.preproc_input):
        pre = np.load(args.preproc_input)
        if "bin_size" in pre:
            bin_size = float(pre["bin_size"])

    np.savez_compressed(
        "data/processed/rnn_latent.npz",
        Z_true=Z,
        Z_pred_teacher=Z_pred_teacher,
        Z_rollout=Z_rollout,
        train_loss_curve=train_log["loss"],
        d_latent=args.d_latent,
        bin_size=bin_size,
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    n_trials = min(20, Z.shape[0])
    for i in range(n_trials):
        ax.plot(Z[i, :, 0], Z[i, :, 1], color="0.7", alpha=0.6)
        ax.plot(Z_rollout[i, :, 0], Z_rollout[i, :, 1], color="tab:blue", alpha=0.7)
    ax.set_title("Latent RNN: true (gray) vs rollout (blue)")
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "rnn_latent_traj.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
