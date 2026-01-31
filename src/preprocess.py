from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class BinnedSpikes:
    """Container for binned spike rate data aligned to behavioral events.

    Attributes
    ----------
    rates : np.ndarray
        Firing rates in Hz, shape (trials, units, bins)
    bin_edges : np.ndarray
        Time bin edges relative to alignment time, shape (bins + 1,)
    align_times : np.ndarray
        Absolute alignment time for each trial, shape (trials,)
    unit_ids : np.ndarray
        Identifiers for each unit, shape (units,)
    trial_mask : np.ndarray
        Boolean mask indicating valid trials, shape (trials,)
    """
    rates: np.ndarray  # shape: (trials, units, bins)
    bin_edges: np.ndarray  # shape: (bins + 1,)
    align_times: np.ndarray  # shape: (trials,)
    unit_ids: np.ndarray  # shape: (units,)
    trial_mask: np.ndarray  # shape: (trials,)


def _get_unit_spike_times(nwb) -> Tuple[List[np.ndarray], np.ndarray]:
    spike_times = nwb.units["spike_times"]
    if hasattr(spike_times, "target"):
        data = np.asarray(spike_times.target.data[:], dtype=float)
        index = np.asarray(spike_times.data[:], dtype=int)
    else:
        data = np.asarray(spike_times.data[:], dtype=float)
        index = None
        try:
            index = np.asarray(spike_times.index.data[:], dtype=int)
        except Exception:
            try:
                index = np.asarray(spike_times.index[:], dtype=int)
            except Exception:
                index = np.asarray(spike_times.index, dtype=int)
    unit_ids = np.asarray(nwb.units.id[:])

    times: List[np.ndarray] = []
    start = 0
    for stop in index:
        times.append(data[start:stop])
        start = stop
    return times, unit_ids


def _get_align_times(nwb, align_col: str) -> np.ndarray:
    trials = nwb.trials
    if align_col not in trials.colnames:
        raise KeyError(f"align_col '{align_col}' not in trials columns")
    return np.asarray(trials[align_col][:], dtype=float)


def _trial_bounds(nwb) -> Tuple[np.ndarray, np.ndarray]:
    trials = nwb.trials
    start = np.asarray(trials["start_time"][:], dtype=float)
    stop = np.asarray(trials["stop_time"][:], dtype=float)
    return start, stop


def bin_spikes(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(float)


def make_trial_tensor(
    nwb,
    bin_size: float = 0.02,
    t_before: float = 0.0,
    t_after: float = 0.6,
    align_col: str = "start_time",
    respect_trial_bounds: bool = True,
    align_times: Optional[np.ndarray] = None,
) -> BinnedSpikes:
    """Bin spike times into trial-aligned firing rate tensor.

    Extracts spike times from NWB file and bins them relative to a behavioral
    alignment event (e.g., movement onset, cue presentation). Computes firing
    rates in Hz for each trial, unit, and time bin.

    Parameters
    ----------
    nwb : NWBFile
        NWB file object containing units and trials tables
    bin_size : float, optional
        Time bin width in seconds (default: 0.02 = 20ms)
    t_before : float, optional
        Time before alignment event to include, in seconds (default: 0.0)
    t_after : float, optional
        Time after alignment event to include, in seconds (default: 0.6)
    align_col : str, optional
        Column name in nwb.trials to use for alignment (default: "start_time")
    respect_trial_bounds : bool, optional
        If True, exclude trials where time window extends beyond trial bounds
        (default: True)
    align_times : np.ndarray, optional
        Custom alignment times to use instead of nwb.trials[align_col].
        Must have same length as number of trials. (default: None)

    Returns
    -------
    BinnedSpikes
        Container with rates array (trials, units, bins) and metadata
    """
    unit_times, unit_ids = _get_unit_spike_times(nwb)
    if align_times is None:
        align_times = _get_align_times(nwb, align_col)
    else:
        align_times = np.asarray(align_times, dtype=float)
    trial_start, trial_stop = _trial_bounds(nwb)
    if align_times.shape[0] != trial_start.shape[0]:
        raise ValueError("align_times must have same length as nwb.trials")

    bin_edges = np.arange(-t_before, t_after + bin_size, bin_size)
    n_trials = align_times.shape[0]
    n_units = len(unit_times)
    n_bins = bin_edges.size - 1

    rates = np.zeros((n_trials, n_units, n_bins), dtype=float)
    trial_mask = np.ones(n_trials, dtype=bool)

    for i, t0 in enumerate(align_times):
        win_start = t0 - t_before
        win_stop = t0 + t_after
        if respect_trial_bounds:
            if win_start < trial_start[i] or win_stop > trial_stop[i]:
                trial_mask[i] = False
                continue

        edges_abs = bin_edges + t0
        for u, st in enumerate(unit_times):
            rates[i, u, :] = bin_spikes(st, edges_abs) / bin_size

    return BinnedSpikes(
        rates=rates,
        bin_edges=bin_edges,
        align_times=align_times,
        unit_ids=unit_ids,
        trial_mask=trial_mask,
    )


def gaussian_smooth(
    rates: np.ndarray,
    sigma_bins: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to firing rates along time axis.

    Uses scipy.ndimage.gaussian_filter1d if available, otherwise falls back
    to manual convolution with a Gaussian kernel.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate array, shape (trials, units, bins)
    sigma_bins : float, optional
        Gaussian kernel standard deviation in bins (default: 1.0).
        If <= 0, returns unsmoothed rates.

    Returns
    -------
    np.ndarray
        Smoothed firing rates, same shape as input
    """
    if sigma_bins <= 0:
        return rates
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(rates, sigma=sigma_bins, axis=-1, mode="nearest")
    except Exception:
        width = int(np.ceil(6 * sigma_bins))
        x = np.arange(-width, width + 1)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= kernel.sum()
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, rates)


def zscore_units(rates: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize firing rates for each unit across all trials and time.

    Subtracts mean and divides by standard deviation computed across the
    trial and time dimensions for each unit independently.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate array, shape (trials, units, bins)
    eps : float, optional
        Small constant added to std to prevent division by zero (default: 1e-8)

    Returns
    -------
    np.ndarray
        Z-scored firing rates, shape (trials, units, bins)
    """
    mean = rates.mean(axis=(0, 2), keepdims=True)
    std = rates.std(axis=(0, 2), keepdims=True)
    return (rates - mean) / (std + eps)


def filter_low_firing_units(
    rates: np.ndarray,
    min_rate_hz: float = 1.0,
    keep_mask_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter out units with low mean firing rates.

    Removes units whose average firing rate across all trials and time bins
    falls below the specified threshold. Low-firing units often have poor
    signal-to-noise ratio and can negatively impact dimensionality reduction.

    Parameters
    ----------
    rates : np.ndarray
        Firing rate array, shape (trials, units, bins)
    min_rate_hz : float, optional
        Minimum mean firing rate threshold in Hz (default: 1.0)
    keep_mask_only : bool, optional
        If True, return unfiltered rates with mask; if False, return
        filtered rates (default: False)

    Returns
    -------
    rates_out : np.ndarray
        Either filtered rates (trials, kept_units, bins) or original rates
    keep_mask : np.ndarray
        Boolean mask indicating which units were kept, shape (units,)
    """
    mean_rate = rates.mean(axis=(0, 2))
    keep = mean_rate >= min_rate_hz
    if keep_mask_only:
        return rates, keep
    return rates[:, keep, :], keep


def preprocess_pipeline(
    nwb,
    bin_size: float = 0.02,
    t_before: float = 0.0,
    t_after: float = 0.6,
    align_col: str = "start_time",
    smooth_sigma_bins: float = 1.0,
    min_rate_hz: float = 1.0,
    align_times: Optional[np.ndarray] = None,
) -> BinnedSpikes:
    """Complete preprocessing pipeline: bin → smooth → filter → normalize.

    Standard preprocessing for population neural analysis:
    1. Bin spikes into trial-aligned firing rates
    2. Apply Gaussian smoothing along time axis
    3. Filter out low-firing units
    4. Z-score normalize each unit

    This produces cleaned, normalized population activity suitable for
    dimensionality reduction (PCA, jPCA, etc.).

    Parameters
    ----------
    nwb : NWBFile
        NWB file object containing units and trials tables
    bin_size : float, optional
        Time bin width in seconds (default: 0.02 = 20ms)
    t_before : float, optional
        Time before alignment event, in seconds (default: 0.0)
    t_after : float, optional
        Time after alignment event, in seconds (default: 0.6)
    align_col : str, optional
        Column in nwb.trials for alignment (default: "start_time")
    smooth_sigma_bins : float, optional
        Gaussian smoothing kernel width in bins (default: 1.0)
    min_rate_hz : float, optional
        Minimum firing rate threshold for unit inclusion (default: 1.0 Hz)
    align_times : np.ndarray, optional
        Custom alignment times (default: None, uses align_col)

    Returns
    -------
    BinnedSpikes
        Preprocessed firing rates with shape (valid_trials, kept_units, bins).
        Only includes trials that fit within trial bounds and units above
        min_rate_hz threshold.
    """
    binned = make_trial_tensor(
        nwb,
        bin_size=bin_size,
        t_before=t_before,
        t_after=t_after,
        align_col=align_col,
        align_times=align_times,
    )
    rates = binned.rates[binned.trial_mask]
    rates = gaussian_smooth(rates, sigma_bins=smooth_sigma_bins)
    rates, keep = filter_low_firing_units(rates, min_rate_hz=min_rate_hz)
    rates = zscore_units(rates)

    return BinnedSpikes(
        rates=rates,
        bin_edges=binned.bin_edges,
        align_times=binned.align_times[binned.trial_mask],
        unit_ids=binned.unit_ids[keep],
        trial_mask=binned.trial_mask,
    )
