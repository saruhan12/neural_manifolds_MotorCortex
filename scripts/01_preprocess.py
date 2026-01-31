from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
from pynwb import NWBHDF5IO

from src.preprocess import preprocess_pipeline


def _get_timeseries_timestamps(ts) -> np.ndarray:
    if ts.timestamps is not None:
        return np.asarray(ts.timestamps[:], dtype=float)
    if ts.starting_time is None or ts.rate is None:
        raise ValueError("TimeSeries missing timestamps and starting_time/rate")
    n = ts.data.shape[0]
    return ts.starting_time + np.arange(n, dtype=float) / ts.rate


def movement_onset_times(
    nwb,
    speed_threshold: float = 0.05,
) -> np.ndarray:
    behavior = nwb.processing["behavior"]
    ts = behavior["finger_vel"]
    data = np.asarray(ts.data[:], dtype=float)
    if data.ndim == 1:
        speed = np.abs(data)
    else:
        speed = np.linalg.norm(data, axis=1)
    times = _get_timeseries_timestamps(ts)

    starts = np.asarray(nwb.trials["start_time"][:], dtype=float)
    stops = np.asarray(nwb.trials["stop_time"][:], dtype=float)

    align = np.copy(starts)
    for i, (t0, t1) in enumerate(zip(starts, stops)):
        in_trial = (times >= t0) & (times <= t1)
        idx = np.where(in_trial & (speed > speed_threshold))[0]
        if idx.size > 0:
            align[i] = times[idx[0]]
    return align


def _resolve_window(
    t_before: Optional[float],
    t_after: Optional[float],
    align_event: str,
) -> tuple[float, float]:
    if t_before is not None and t_after is not None:
        return t_before, t_after
    if align_event == "movement_onset":
        return 0.2 if t_before is None else t_before, 0.6 if t_after is None else t_after
    return 0.1 if t_before is None else t_before, 0.5 if t_after is None else t_after


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MC_RTT into binned rates.")
    parser.add_argument("nwb_path", help="Path to NWB file")
    parser.add_argument(
        "--output",
        default="data/processed/preprocessed.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--align",
        choices=["start_time", "movement_onset"],
        default="movement_onset",
        help="Alignment event for binning",
    )
    parser.add_argument("--bin-size", type=float, default=0.02)
    parser.add_argument("--t-before", type=float, default=None)
    parser.add_argument("--t-after", type=float, default=None)
    parser.add_argument("--smooth-sigma-bins", type=float, default=1.0)
    parser.add_argument("--min-rate-hz", type=float, default=1.0)
    parser.add_argument("--speed-threshold", type=float, default=0.05)
    args = parser.parse_args()

    t_before, t_after = _resolve_window(args.t_before, args.t_after, args.align)

    with NWBHDF5IO(args.nwb_path, "r") as io:
        nwb = io.read()
        if args.align == "movement_onset":
            align_times = movement_onset_times(nwb, speed_threshold=args.speed_threshold)
        else:
            align_times = None

        binned = preprocess_pipeline(
            nwb,
            bin_size=args.bin_size,
            t_before=t_before,
            t_after=t_after,
            align_col="start_time",
            smooth_sigma_bins=args.smooth_sigma_bins,
            min_rate_hz=args.min_rate_hz,
            align_times=align_times,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        rates=binned.rates,
        bin_edges=binned.bin_edges,
        align_times=binned.align_times,
        unit_ids=binned.unit_ids,
        trial_mask=binned.trial_mask,
        bin_size=args.bin_size,
        t_before=t_before,
        t_after=t_after,
        align_event=args.align,
    )


if __name__ == "__main__":
    main()
