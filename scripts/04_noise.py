from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.noise import noise_robustness_pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise robustness analysis for PCA subspace.")
    parser.add_argument(
        "--input",
        default="data/processed/preprocessed.npz",
        help="Path to preprocessed .npz",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--plot-dir", default="report/figs")
    args = parser.parse_args()

    data = np.load(args.input)
    rates = data["rates"]
    res = noise_robustness_pca(rates, n_components=args.n_components, snr_db=args.snr_db)

    os.makedirs(args.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    pcs = np.arange(1, res.explained_variance_ratio.size + 1)
    ax.plot(pcs, res.explained_variance_ratio, marker="o", label="Clean")
    ax.plot(pcs, res.noisy_explained_variance_ratio, marker="s", label="Noisy")
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance ratio")
    ax.set_title(f"Noise robustness (SNR {res.snr_db:.1f} dB)\nSubspace similarity {res.subspace_similarity:.2f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "noise_robustness.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
