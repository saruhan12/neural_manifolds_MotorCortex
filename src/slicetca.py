from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class SliceTCAResult:
    """Results from sliceTCA tensor decomposition.

    sliceTCA decomposes 3D neural tensors (trials × units × time) into
    low-rank components, useful for denoising and extracting shared structure.

    Attributes
    ----------
    recon : np.ndarray
        Reconstructed tensor from components, shape (trials, units, time)
    components : Dict[str, np.ndarray]
        Dictionary of component arrays extracted by sliceTCA
    n_components : Tuple[int, int, int]
        Number of components per mode (trials, units, time)
    loss : Any, optional
        Final reconstruction loss if available
    """
    recon: np.ndarray
    components: Dict[str, np.ndarray]
    n_components: Tuple[int, int, int]
    loss: Any | None


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _init_with_signature(cls, **kwargs):
    import inspect

    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered)


def _call_with_signature(func, **kwargs):
    import inspect

    sig = inspect.signature(func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**filtered)


def _get_components(model) -> Dict[str, np.ndarray]:
    if hasattr(model, "get_components"):
        comps = _call_with_signature(model.get_components, numpy=True, detach=False)
        if isinstance(comps, dict):
            return {str(k): np.asarray(v) for k, v in comps.items()}
        if isinstance(comps, (list, tuple)):
            return _pack_components(comps)
        return {"component_0": np.asarray(comps)}
    for name in ("components_", "components", "factors_", "factors"):
        if hasattr(model, name):
            val = getattr(model, name)
            if isinstance(val, dict):
                return {str(k): np.asarray(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return _pack_components(val)
            return {"component_0": np.asarray(val)}
    return {}


def _pack_components(components) -> Dict[str, np.ndarray]:
    if isinstance(components, dict):
        return {str(k): np.asarray(v) for k, v in components.items()}
    if isinstance(components, (list, tuple)):
        packed: Dict[str, np.ndarray] = {}
        for i, item in enumerate(components):
            if isinstance(item, (list, tuple)):
                for j, arr in enumerate(item):
                    packed[f"component_{i}_{j}"] = np.asarray(arr)
            else:
                packed[f"component_{i}"] = np.asarray(item)
        return packed
    if components is None:
        return {}
    return {"component_0": np.asarray(components)}


def _extract_recon(
    model,
    out,
    shape: tuple[int, int, int],
    X: np.ndarray,
) -> np.ndarray | None:
    if isinstance(out, np.ndarray) and out.shape == shape:
        return out
    if isinstance(out, dict):
        for key in ("recon", "reconstruction", "X_hat", "Xhat"):
            if key in out and isinstance(out[key], np.ndarray) and out[key].shape == shape:
                return out[key]
    for name in ("reconstruction_", "recon_", "reconstructed_", "Xhat_", "X_hat_"):
        if hasattr(model, name):
            recon = getattr(model, name)
            if isinstance(recon, np.ndarray) and recon.shape == shape:
                return recon
    for method in ("reconstruct", "predict", "forward"):
        if hasattr(model, method):
            try:
                func = getattr(model, method)
                recon = _call_with_signature(func, data=X, X=X)
                if isinstance(recon, np.ndarray) and recon.shape == shape:
                    return recon
            except Exception:
                pass
    return None


def _extract_loss(model, out):
    if isinstance(out, dict):
        for key in ("loss", "objective", "metric"):
            if key in out:
                return out[key]
    for name in ("loss_", "loss", "objective_", "objective"):
        if hasattr(model, name):
            return getattr(model, name)
    return None


def _apply_invariance(model) -> None:
    try:
        from slicetca._invariance.analytic_invariance import svd_basis

        svd_basis(model)
    except Exception:
        pass


def _reconstruct_from_components(model, n_components: Tuple[int, int, int]) -> np.ndarray | None:
    if not hasattr(model, "construct_single_component"):
        return None
    try:
        import torch

        recon = None
        for partition, count in enumerate(n_components):
            for k in range(count):
                comp = model.construct_single_component(partition, k)
                if recon is None:
                    recon = comp
                else:
                    recon = recon + comp
        if recon is None:
            return None
        if isinstance(recon, torch.Tensor):
            recon = recon.detach().cpu().numpy()
        return np.asarray(recon, dtype=float)
    except Exception:
        return None


def fit_slicetca(
    rates: np.ndarray,
    n_components: Tuple[int, int, int] = (3, 3, 1),
    n_inits: int = 3,
    seed: int = 0,
    device: str = "cpu",
) -> SliceTCAResult:
    """Fit sliceTCA tensor decomposition for denoising and structure extraction.

    sliceTCA (Williams et al. 2018) performs low-rank tensor decomposition
    on (trials × units × time) data. This can denoise data and extract
    shared trial-to-trial and temporal structure.

    This is a generic wrapper that adapts to different sliceTCA API versions.

    Parameters
    ----------
    rates : np.ndarray
        Neural tensor, shape (trials, units, time)
    n_components : Tuple[int, int, int], optional
        Rank for each mode (trials, units, time) (default: (3, 3, 1))
    n_inits : int, optional
        Number of random initializations to try (default: 3)
    seed : int, optional
        Random seed (default: 0)
    device : str, optional
        Device for computation: "cpu" or "cuda" (default: "cpu")

    Returns
    -------
    SliceTCAResult
        Reconstruction, components, and loss

    References
    ----------
    Williams AH, Kunz E, Kornblith S, Linderman SW (2018). Generalized
    shape metrics on neural representations. NeurIPS.
    """
    _set_seed(seed)
    X = np.asarray(rates, dtype=float)
    if X.ndim != 3:
        raise ValueError("rates must be 3D: (trials, units, time)")

    model = None
    out = None
    components_raw = None

    try:
        from slicetca import decompose

        out = _call_with_signature(
            decompose,
            data=X,
            X=X,
            n_components=n_components,
            number_components=n_components,
            ranks=n_components,
            n_inits=n_inits,
            seed=seed,
            device=device,
            max_iter=1000,
            progress_bar=False,
            verbose=False,
        )
        if isinstance(out, tuple) and len(out) == 2:
            components_raw, model = out
        elif isinstance(out, dict):
            if "model" in out:
                model = out["model"]
            if "components" in out:
                components_raw = out["components"]
    except Exception:
        out = None

    if model is None:
        raise ImportError(
            "sliceTCA high-level API unavailable. Ensure slicetca provides decompose()."
        )

    _apply_invariance(model)

    recon = _extract_recon(model, out, X.shape, X)
    if recon is None:
        recon = _reconstruct_from_components(model, n_components)
    if recon is None:
        raise RuntimeError("Could not extract reconstruction from sliceTCA output.")
    recon = np.asarray(recon, dtype=float)

    components = _get_components(model)
    if not components:
        components = _pack_components(components_raw)
    loss = _extract_loss(model, out)

    return SliceTCAResult(
        recon=recon,
        components=components,
        n_components=n_components,
        loss=loss,
    )
