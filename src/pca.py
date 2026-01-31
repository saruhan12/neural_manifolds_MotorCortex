from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    """Results from Principal Component Analysis on neural population activity.

    Attributes
    ----------
    components : np.ndarray
        Principal component vectors, shape (n_components, units)
    explained_variance : np.ndarray
        Variance explained by each PC, shape (n_components,)
    explained_variance_ratio : np.ndarray
        Fraction of total variance explained by each PC, shape (n_components,)
    scores : np.ndarray
        Neural trajectories projected onto PCs, shape (trials, bins, n_components)
    mean : np.ndarray
        Mean firing rate across all timepoints, shape (units,)
    """
    components: np.ndarray  # shape: (n_components, units)
    explained_variance: np.ndarray  # shape: (n_components,)
    explained_variance_ratio: np.ndarray  # shape: (n_components,)
    scores: np.ndarray  # shape: (trials, bins, n_components)
    mean: np.ndarray  # shape: (units,)


def _flatten_trials(rates: np.ndarray) -> np.ndarray:
    trials, units, bins = rates.shape
    return rates.transpose(0, 2, 1).reshape(trials * bins, units)


def fit_pca(
    rates: np.ndarray,
    n_components: int = 10,
    whiten: bool = False,
    random_state: Optional[int] = 0,
) -> PCAResult:
    """Fit PCA on neural population activity to extract low-dimensional structure.

    Flattens trial and time dimensions, fits PCA, then reshapes scores back to
    (trials, bins, n_components) to preserve temporal structure. This reveals
    the dominant patterns of population covariation.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate tensor, shape (trials, units, bins)
    n_components : int, optional
        Number of principal components to extract (default: 10)
    whiten : bool, optional
        Whether to whiten the components (default: False)
    random_state : int, optional
        Random seed for reproducibility (default: 0)

    Returns
    -------
    PCAResult
        Container with PC components, explained variance, and projected scores
    """
    X = _flatten_trials(rates)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    scores = pca.fit_transform(X)
    trials, units, bins = rates.shape
    scores = scores.reshape(trials, bins, n_components)
    return PCAResult(
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        scores=scores,
        mean=pca.mean_,
    )


def project_pca(
    rates: np.ndarray,
    components: np.ndarray,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project new neural data onto previously fitted PCA components.

    Useful for applying a PCA model fitted on training data to test data,
    or for projecting data from different experimental conditions onto the
    same low-dimensional subspace for comparison.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate tensor, shape (trials, units, bins)
    components : np.ndarray
        PCA component vectors, shape (n_components, units)
    mean : np.ndarray, optional
        Mean to subtract before projection, shape (units,).
        If None, data is not centered (default: None)

    Returns
    -------
    np.ndarray
        Projected scores, shape (trials, bins, n_components)
    """
    X = _flatten_trials(rates)
    if mean is not None:
        X = X - mean
    scores = X @ components.T
    trials, units, bins = rates.shape
    return scores.reshape(trials, bins, components.shape[0])


def variance_explained(
    explained_variance_ratio: np.ndarray,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute individual and cumulative variance explained by PCs.

    Helper function for visualizing how much of the population variance
    is captured by the top N principal components.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Variance ratio for each PC, shape (n_pcs,)
    n_components : int, optional
        Number of components to include. If None, uses all (default: None)

    Returns
    -------
    explained : np.ndarray
        Individual variance ratios, shape (n_components,)
    cumulative : np.ndarray
        Cumulative variance ratios, shape (n_components,)
    """
    if n_components is not None:
        explained_variance_ratio = explained_variance_ratio[:n_components]
    cumulative = np.cumsum(explained_variance_ratio)
    return explained_variance_ratio, cumulative
