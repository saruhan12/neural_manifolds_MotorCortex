# Low-Dimensional Neural Manifolds and Noise Robustness in Motor Cortex

Minimal analysis pipeline for DANDI `000129` (MC_RTT) to study low-dimensional population dynamics, jPCA rotations, and subspace robustness to noise. Focus is interpretability over benchmarking.

## Scope

- Dataset: MC_RTT NWB files already downloaded
- Core analyses: preprocessing, PCA, jPCA, sliceTCA
- Outputs: processed data and report figures

## Layout

```
.
├── scripts/     # CLI entry points
├── src/         # Core analysis code
```

Key code:

- `src/preprocess.py` trial alignment + binning + smoothing + z-scoring
- `src/pca.py` PCA fit/projection
- `src/jpca.py` jPCA fit (linear dynamics + skew-symmetric plane)
- `src/noise.py` noise injection + subspace metric
- `src/viz.py` plotting helpers

## Setup

Recommended stack:

- Python 3.10+
- NumPy, SciPy, scikit-learn, matplotlib
- PyNWB

Conda example:

```bash
conda create -n rnn_winter numpy scipy scikit-learn matplotlib pynwb
conda activate rnn_winter
```

## Data

Expected paths:

```
000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb
000129/sub-Indy/sub-Indy_desc-test_ecephys.nwb
```

## Pipeline

1. **Preprocess**: bin spikes, smooth, z-score, filter low-rate units
2. **PCA**: fit and project trial trajectories
3. **jPCA**: fit linear dynamics and extract rotational plane
4. **Noise robustness**: compare clean vs noisy PCA subspaces

Optional stages exist for sliceTCA denoising and latent RNNs; see `scripts/05_slicetca.py` onward.

## Run

Preprocess NWB to `data/processed/preprocessed.npz`:

```bash
python scripts/01_preprocess.py 000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb
```

PCA and plots:

```bash
python scripts/02_pca.py
```

jPCA and plots:

```bash
python scripts/03_jpca.py
```

Noise robustness:

```bash
python scripts/04_noise.py --snr-db 10
```

## Notes

- Default alignment is movement onset; use `--align start_time` to change.
- Window sizes (`t_before`, `t_after`) and `min_rate_hz` materially affect results.
- Interpret results as dynamical-geometry evidence, not predictive performance.

## Citation

```
O'Doherty, Joseph (2024) MC_RTT: macaque motor cortex spiking activity during self-paced reaching
(Version 0.241017.1444) [Data set]. DANDI archive. https://doi.org/10.48324/dandi.000129/0.241017.1444
```
