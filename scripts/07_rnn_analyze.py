from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.noise import subspace_similarity


def _fit_jpca_latent(Z: np.ndarray, dt: float, plane_index: int = 0):
    if Z.ndim != 3:
        raise ValueError("Z must be 3D: (trials, time, d_latent)")
    trials, time, d_latent = Z.shape
    if d_latent < 2:
        raise ValueError("Need at least 2 latent dims for jPCA")

    Xdot = np.gradient(Z, dt, axis=1)
    X = Z.reshape(-1, d_latent)
    Xdot_flat = Xdot.reshape(-1, d_latent)
    M, _, _, _ = np.linalg.lstsq(X, Xdot_flat, rcond=None)
    M = M.T
    M_skew = 0.5 * (M - M.T)

    evals, evecs = np.linalg.eig(M_skew)
    imag = np.imag(evals)
    order = np.argsort(-np.abs(imag))
    evals = evals[order]
    evecs = evecs[:, order]

    pair_start = 2 * plane_index
    if pair_start + 1 >= evecs.shape[1]:
        raise ValueError("plane_index out of range for jPCA planes")
    plane = np.real(evecs[:, pair_start : pair_start + 2].T)

    scores_2d = Z @ plane.T
    return {
        "components": plane,
        "eigenvalues": evals,
        "scores": scores_2d,
        "skew_matrix": M_skew,
    }


def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    QA, _ = np.linalg.qr(A.T)
    QB, _ = np.linalg.qr(B.T)
    svals = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return svals


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latent RNN rollouts with jPCA.")
    parser.add_argument("--input", default="data/processed/rnn_latent.npz")
    parser.add_argument("--output", default="data/processed/rnn_jpca.npz")
    parser.add_argument("--plane-index", type=int, default=0)
    parser.add_argument("--plot-dir", default="report/figs")
    args = parser.parse_args()

    data = np.load(args.input)
    Z_true = data["Z_true"]
    Z_rollout = data["Z_rollout"]
    bin_size = float(data["bin_size"]) if "bin_size" in data else 1.0
    dt = bin_size if np.isfinite(bin_size) and bin_size > 0 else 1.0

    jpca = _fit_jpca_latent(Z_rollout, dt=dt, plane_index=args.plane_index)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        components=jpca["components"],
        eigenvalues=jpca["eigenvalues"],
        scores=jpca["scores"],
        skew_matrix=jpca["skew_matrix"],
        dt=dt,
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    n_trials = min(20, Z_rollout.shape[0])
    for i in range(n_trials):
        ax.plot(jpca["scores"][i, :, 0], jpca["scores"][i, :, 1], alpha=0.7)
    ax.set_title("RNN rollout jPCA plane")
    ax.set_xlabel("jPC1")
    ax.set_ylabel("jPC2")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "rnn_jpca_plane.png"), dpi=150)
    plt.close(fig)

    d = min(Z_true.shape[2], Z_rollout.shape[2])
    pca_true = PCA(n_components=d).fit(Z_true.reshape(-1, Z_true.shape[2]))
    pca_roll = PCA(n_components=d).fit(Z_rollout.reshape(-1, Z_rollout.shape[2]))
    svals = _principal_angles(pca_true.components_, pca_roll.components_)
    similarity = subspace_similarity(pca_true.components_, pca_roll.components_)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, svals.size + 1), svals, marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("cos(angle)")
    ax.set_title(f"Data vs RNN subspace similarity: {similarity:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "rnn_vs_data_angles.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
