from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class NoiseResult:
    """Results from noise robustness analysis on PCA subspaces.

    Attributes
    ----------
    snr_db : float
        Signal-to-noise ratio in decibels used for noise injection
    subspace_similarity : float
        Mean cosine of principal angles between clean and noisy PCA subspaces.
        Range [0, 1], where 1 = identical subspaces.
    explained_variance_ratio : np.ndarray
        Variance explained by each PC in clean data, shape (n_components,)
    noisy_explained_variance_ratio : np.ndarray
        Variance explained by each PC in noisy data, shape (n_components,)
    """
    snr_db: float
    subspace_similarity: float
    explained_variance_ratio: np.ndarray
    noisy_explained_variance_ratio: np.ndarray


def add_gaussian_noise(
    rates: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise to neural data at specified signal-to-noise ratio.

    Noise power is calibrated relative to signal power to achieve the target
    SNR in decibels: SNR_dB = 10 * log10(signal_power / noise_power)

    Parameters
    ----------
    rates : np.ndarray
        Clean firing rate tensor, shape (trials, units, bins)
    snr_db : float
        Target signal-to-noise ratio in decibels. Higher = less noise.
        Typical values: 5-20 dB.
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default: seeded with 0)

    Returns
    -------
    np.ndarray
        Noisy firing rates, same shape as input
    """
    if rng is None:
        rng = np.random.default_rng(0)
    signal_power = np.mean(rates**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power), size=rates.shape)
    return rates + noise


def _flatten_trials(rates: np.ndarray) -> np.ndarray:
    trials, units, bins = rates.shape
    return rates.transpose(0, 2, 1).reshape(trials * bins, units)


def _subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    QA, _ = np.linalg.qr(A.T)
    QB, _ = np.linalg.qr(B.T)
    svals = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return float(np.mean(svals))


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Compute similarity between two linear subspaces.

    Calculates the mean cosine of principal angles using SVD of the overlap
    between orthonormal bases. This metric quantifies how closely two subspaces
    align, independent of the coordinate system used to represent them.

    Parameters
    ----------
    A : np.ndarray
        Basis vectors for first subspace, shape (n_components, n_features)
    B : np.ndarray
        Basis vectors for second subspace, shape (n_components, n_features)

    Returns
    -------
    float
        Similarity metric in [0, 1]. Value of 1 means identical subspaces,
        0 means orthogonal subspaces.
    """
    return _subspace_similarity(A, B)


def noise_robustness_pca(
    rates: np.ndarray,
    n_components: int = 10,
    snr_db: float = 10.0,
    random_state: int = 0,
) -> NoiseResult:
    """Test robustness of PCA subspace to additive Gaussian noise.

    Compares PCA subspaces extracted from clean vs. noisy neural data to
    assess how stable the low-dimensional representation is under noise
    perturbations. Robust subspaces indicate that the extracted structure
    reflects true population dynamics rather than noise artifacts.

    Parameters
    ----------
    rates : np.ndarray
        Clean firing rate tensor, shape (trials, units, bins)
    n_components : int, optional
        Number of principal components to extract and compare (default: 10)
    snr_db : float, optional
        Signal-to-noise ratio for noise injection in dB (default: 10.0)
    random_state : int, optional
        Random seed for reproducibility (default: 0)

    Returns
    -------
    NoiseResult
        Container with subspace similarity metric and variance explained
        for both clean and noisy data
    """
    X = _flatten_trials(rates)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)

    noisy = add_gaussian_noise(rates, snr_db=snr_db, rng=np.random.default_rng(random_state))
    Xn = _flatten_trials(noisy)
    pca_noisy = PCA(n_components=n_components, random_state=random_state)
    pca_noisy.fit(Xn)

    similarity = _subspace_similarity(pca.components_, pca_noisy.components_)
    return NoiseResult(
        snr_db=snr_db,
        subspace_similarity=similarity,
        explained_variance_ratio=pca.explained_variance_ratio_,
        noisy_explained_variance_ratio=pca_noisy.explained_variance_ratio_,
    )
