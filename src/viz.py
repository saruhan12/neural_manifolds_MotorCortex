from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_variance_explained(
    explained: np.ndarray,
    cumulative: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot individual and cumulative variance explained by PCs.

    Parameters
    ----------
    explained : np.ndarray
        Variance ratio per component
    cumulative : np.ndarray, optional
        Cumulative variance ratio
    ax : plt.Axes, optional
        Axes to plot on (creates new if None)

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, explained.size + 1), explained, marker="o", label="Explained")
    if cumulative is not None:
        ax.plot(np.arange(1, cumulative.size + 1), cumulative, marker="s", label="Cumulative")
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA variance explained")
    ax.legend(frameon=False)
    return ax


def plot_trajectories_2d(
    scores: np.ndarray,
    trial_ids: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.6,
) -> plt.Axes:
    """Plot neural trajectories in 2D latent space.

    Parameters
    ----------
    scores : np.ndarray
        Trajectories in 2D, shape (trials, bins, 2)
    trial_ids : Sequence[int], optional
        Which trials to plot (default: all)
    ax : plt.Axes, optional
        Axes to plot on (creates new if None)
    alpha : float, optional
        Line transparency (default: 0.6)

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    trials = scores.shape[0]
    if trial_ids is None:
        trial_ids = range(trials)
    for i in trial_ids:
        ax.plot(scores[i, :, 0], scores[i, :, 1], alpha=alpha)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_jpca_plane(
    scores: np.ndarray,
    trial_ids: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot neural trajectories in jPCA rotational plane.

    Convenience wrapper around plot_trajectories_2d with jPCA-specific title.

    Parameters
    ----------
    scores : np.ndarray
        jPCA trajectories, shape (trials, bins, 2)
    trial_ids : Sequence[int], optional
        Which trials to plot (default: all)
    ax : plt.Axes, optional
        Axes to plot on (creates new if None)

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    ax = plot_trajectories_2d(scores, trial_ids=trial_ids, ax=ax)
    ax.set_title("jPCA plane trajectories")
    return ax
