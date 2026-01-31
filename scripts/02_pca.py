from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.pca import fit_pca, variance_explained
from src.viz import plot_variance_explained, plot_trajectories_2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA on preprocessed rates.")
    parser.add_argument(
        "--input",
        default="data/processed/preprocessed.npz",
        help="Path to preprocessed .npz",
    )
    parser.add_argument(
        "--output",
        default="data/processed/pca.npz",
        help="Output .npz for PCA results",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--plot-dir", default="report/figs")
    args = parser.parse_args()

    data = np.load(args.input)
    rates = data["rates"]
    pca_res = fit_pca(rates, n_components=args.n_components)
    explained, cumulative = variance_explained(pca_res.explained_variance_ratio)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        components=pca_res.components,
        explained_variance=pca_res.explained_variance,
        explained_variance_ratio=pca_res.explained_variance_ratio,
        scores=pca_res.scores,
        mean=pca_res.mean,
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_variance_explained(explained, cumulative=cumulative, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "pca_variance.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_trajectories_2d(pca_res.scores[:, :, :2], trial_ids=range(min(20, rates.shape[0])), ax=ax)
    ax.set_title("PCA trajectories (first 2 PCs)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "pca_traj.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
