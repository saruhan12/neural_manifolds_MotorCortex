from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class JPCAResult:
    """Results from jPCA (rotational dynamics) analysis.

    jPCA identifies 2D planes of rotational neural dynamics by finding the
    skew-symmetric component of linear dynamics fitted to PCA-reduced data.
    See Churchland et al. (2012) Nature for mathematical details.

    Attributes
    ----------
    components : np.ndarray
        Basis vectors for the jPCA plane, shape (2, n_pca_dims)
    eigenvalues : np.ndarray
        Eigenvalues of skew-symmetric dynamics matrix, sorted by imaginary magnitude
    scores : np.ndarray
        Neural trajectories projected onto jPCA plane, shape (trials, bins, 2)
    skew_matrix : np.ndarray
        Skew-symmetric component of linear dynamics, shape (n_pca_dims, n_pca_dims)
    pca_components : np.ndarray
        Intermediate PCA basis used before jPCA, shape (n_pca_dims, units)
    pca_mean : np.ndarray
        Mean activity used for PCA centering, shape (units,)
    """
    components: np.ndarray  # jPCA plane components, shape: (2, n_dims)
    eigenvalues: np.ndarray  # eigenvalues of skew-symmetric matrix
    scores: np.ndarray  # projected trajectories, shape: (trials, bins, 2)
    skew_matrix: np.ndarray  # shape: (n_dims, n_dims)
    pca_components: np.ndarray  # shape: (n_dims, units)
    pca_mean: np.ndarray  # shape: (units,)


def _flatten_trials(rates: np.ndarray) -> np.ndarray:
    trials, units, bins = rates.shape
    return rates.transpose(0, 2, 1).reshape(trials * bins, units)


def _reshape_scores(scores: np.ndarray, trials: int, bins: int) -> np.ndarray:
    return scores.reshape(trials, bins, scores.shape[1])


def _time_derivative(scores: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(scores, dt, axis=1)


def _fit_linear_dynamics(X: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
    X_flat = X.reshape(-1, X.shape[-1])
    Xdot_flat = Xdot.reshape(-1, Xdot.shape[-1])
    M, _, _, _ = np.linalg.lstsq(X_flat, Xdot_flat, rcond=None)
    return M.T


def fit_jpca(
    rates: np.ndarray,
    bin_size: float,
    n_pca_dims: int = 6,
    plane_index: int = 0,
) -> JPCAResult:
    """Fit jPCA to identify rotational dynamics in neural population activity.

    jPCA (Churchland et al. 2012) extracts 2D planes exhibiting rotational
    dynamics from population activity. Algorithm:
    1. Reduce dimensionality with PCA (captures ~80-90% of variance)
    2. Compute time derivatives of trajectories
    3. Fit linear dynamics: dx/dt = M*x
    4. Extract skew-symmetric component: M_skew = (M - M^T) / 2
    5. Find plane with largest rotational eigenvalues

    Rotational dynamics are characteristic of motor preparation and movement
    generation in motor cortex.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate tensor, shape (trials, units, bins)
    bin_size : float
        Time bin width in seconds, used for computing derivatives
    n_pca_dims : int, optional
        Number of PCA dimensions to use before jPCA (default: 6).
        Typical values are 6-12.
    plane_index : int, optional
        Which rotational plane to extract. 0 = largest rotation (default: 0).
        Eigenvalues come in conjugate pairs, so plane_index indexes pairs.

    Returns
    -------
    JPCAResult
        Container with jPCA plane, eigenvalues, and projected 2D trajectories

    References
    ----------
    Churchland MM, Cunningham JP, Kaufman MT, et al. (2012). Neural population
    dynamics during reaching. Nature 487:51-56.
    """
    trials, units, bins = rates.shape
    X = _flatten_trials(rates)
    pca = PCA(n_components=n_pca_dims, random_state=0)
    scores = pca.fit_transform(X)
    scores = _reshape_scores(scores, trials, bins)

    Xdot = _time_derivative(scores, bin_size)
    M = _fit_linear_dynamics(scores, Xdot)
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

    scores_2d = scores @ plane.T
    return JPCAResult(
        components=plane,
        eigenvalues=evals,
        scores=scores_2d,
        skew_matrix=M_skew,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
    )


def project_jpca(
    rates: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    jpca_components: np.ndarray,
) -> np.ndarray:
    """Project new neural data onto previously fitted jPCA plane.

    Applies the two-stage projection: first onto PCA subspace, then onto
    the 2D jPCA rotational plane. Useful for comparing different conditions
    or datasets in the same low-dimensional space.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate tensor, shape (trials, units, bins)
    pca_components : np.ndarray
        Intermediate PCA basis, shape (n_pca_dims, units)
    pca_mean : np.ndarray
        Mean for centering, shape (units,)
    jpca_components : np.ndarray
        jPCA plane basis vectors, shape (2, n_pca_dims)

    Returns
    -------
    np.ndarray
        Trajectories in jPCA plane, shape (trials, bins, 2)
    """
    X = _flatten_trials(rates)
    X = X - pca_mean
    scores = X @ pca_components.T
    trials, units, bins = rates.shape
    scores = _reshape_scores(scores, trials, bins)
    return scores @ jpca_components.T
