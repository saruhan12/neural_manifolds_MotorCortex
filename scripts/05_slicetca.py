from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.noise import subspace_similarity
from src.pca import fit_pca, project_pca, variance_explained
from src.slicetca import fit_slicetca


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _infer_factor_by_dim(components: dict, dim: int):
    for key, arr in components.items():
        if arr.ndim == 2 and arr.shape[0] == dim:
            return key, arr
    return None, None


def _plot_components(components: dict, rates_shape: tuple, out_path: str) -> None:
    trials, units, bins = rates_shape
    t_key, t_factor = _infer_factor_by_dim(components, trials)
    n_key, n_factor = _infer_factor_by_dim(components, units)
    b_key, b_factor = _infer_factor_by_dim(components, bins)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    if t_factor is not None:
        axes[0].imshow(t_factor.T, aspect="auto", cmap="viridis")
        axes[0].set_title(f"Trial factors ({t_key})")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Component")
    else:
        axes[0].text(0.5, 0.5, "No trial factor", ha="center", va="center")

    if n_factor is not None:
        axes[1].imshow(n_factor.T, aspect="auto", cmap="viridis")
        axes[1].set_title(f"Neuron factors ({n_key})")
        axes[1].set_xlabel("Unit")
        axes[1].set_ylabel("Component")
    else:
        axes[1].text(0.5, 0.5, "No neuron factor", ha="center", va="center")

    if b_factor is not None:
        axes[2].plot(b_factor)
        axes[2].set_title(f"Time factors ({b_key})")
        axes[2].set_xlabel("Bin")
        axes[2].set_ylabel("Weight")
    else:
        axes[2].text(0.5, 0.5, "No time factor", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_denoise_pca(
    rates: np.ndarray,
    recon: np.ndarray,
    n_components: int,
    out_path: str,
) -> float:
    pca_raw = fit_pca(rates, n_components=n_components)
    pca_rec = fit_pca(recon, n_components=n_components)

    ev_raw, cum_raw = variance_explained(pca_raw.explained_variance_ratio)
    ev_rec, cum_rec = variance_explained(pca_rec.explained_variance_ratio)

    scores_raw = pca_raw.scores[:, :, :2]
    scores_rec = project_pca(recon, pca_raw.components, mean=pca_raw.mean)[:, :, :2]
    traj_raw = scores_raw.mean(axis=0)
    traj_rec = scores_rec.mean(axis=0)

    similarity = subspace_similarity(pca_raw.components, pca_rec.components)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(np.arange(1, ev_raw.size + 1), cum_raw, marker="o", label="Raw")
    axes[0].plot(np.arange(1, ev_rec.size + 1), cum_rec, marker="s", label="sliceTCA")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Cumulative variance")
    axes[0].set_title("PCA variance explained")
    axes[0].legend(frameon=False)

    axes[1].plot(traj_raw[:, 0], traj_raw[:, 1], color="0.4", label="Raw mean")
    axes[1].plot(traj_rec[:, 0], traj_rec[:, 1], color="tab:red", label="sliceTCA mean")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title(f"Mean trajectory (subspace {similarity:.2f})")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return similarity


def _try_slicetca_plot(components: dict, out_path: str) -> bool:
    try:
        from slicetca import plot

        fig = plot(components)
        if hasattr(fig, "savefig"):
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return True
    except Exception:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sliceTCA on preprocessed rates.")
    parser.add_argument(
        "--input",
        default="data/processed/preprocessed.npz",
        help="Path to preprocessed .npz",
    )
    parser.add_argument(
        "--output",
        default="data/processed/slicetca.npz",
        help="Output .npz for sliceTCA results",
    )
    parser.add_argument("--n-components", type=int, nargs=3, default=(3, 3, 1))
    parser.add_argument("--n-inits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot-dir", default="report/figs")
    parser.add_argument("--pca-components", type=int, default=8)
    args = parser.parse_args()

    _set_seed(args.seed)
    data = np.load(args.input)
    rates = data["rates"]

    res = fit_slicetca(
        rates,
        n_components=tuple(args.n_components),
        n_inits=args.n_inits,
        seed=args.seed,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_dict = {
        "rates_recon": res.recon,
        "n_components": np.asarray(res.n_components),
        "loss": res.loss if res.loss is not None else np.nan,
    }
    for key, val in res.components.items():
        save_dict[f"component_{key}"] = val
    np.savez_compressed(args.output, **save_dict)

    os.makedirs(args.plot_dir, exist_ok=True)
    components_path = os.path.join(args.plot_dir, "slicetca_components.png")
    if not _try_slicetca_plot(res.components, components_path):
        _plot_components(res.components, rates.shape, components_path)

    similarity = _plot_denoise_pca(
        rates,
        res.recon,
        n_components=args.pca_components,
        out_path=os.path.join(args.plot_dir, "slicetca_denoise_pca.png"),
    )

    metric_path = os.path.join(args.plot_dir, "slicetca_subspace_metric.txt")
    with open(metric_path, "w", encoding="utf-8") as f:
        f.write(f"Subspace similarity (mean cos angle): {similarity:.4f}\n")


if __name__ == "__main__":
    main()
