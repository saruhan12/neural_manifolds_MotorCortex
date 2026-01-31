from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.jpca import fit_jpca
from src.viz import plot_jpca_plane


def main() -> None:
    parser = argparse.ArgumentParser(description="Run jPCA on preprocessed rates.")
    parser.add_argument(
        "--input",
        default="data/processed/preprocessed.npz",
        help="Path to preprocessed .npz",
    )
    parser.add_argument(
        "--output",
        default="data/processed/jpca.npz",
        help="Output .npz for jPCA results",
    )
    parser.add_argument("--n-pca-dims", type=int, default=6)
    parser.add_argument("--plane-index", type=int, default=0)
    parser.add_argument("--plot-dir", default="report/figs")
    args = parser.parse_args()

    data = np.load(args.input)
    rates = data["rates"]
    bin_size = float(data["bin_size"])

    jpca_res = fit_jpca(
        rates,
        bin_size=bin_size,
        n_pca_dims=args.n_pca_dims,
        plane_index=args.plane_index,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        components=jpca_res.components,
        eigenvalues=jpca_res.eigenvalues,
        scores=jpca_res.scores,
        skew_matrix=jpca_res.skew_matrix,
        pca_components=jpca_res.pca_components,
        pca_mean=jpca_res.pca_mean,
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_jpca_plane(jpca_res.scores, trial_ids=range(min(20, rates.shape[0])), ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "jpca_plane.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
