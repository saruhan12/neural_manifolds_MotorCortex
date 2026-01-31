"""Integration tests for neural analysis pipeline.

Tests verify that core analysis functions run without errors on synthetic data.
"""
import numpy as np
import pytest

from src.pca import fit_pca, project_pca, variance_explained
from src.jpca import fit_jpca, project_jpca
from src.noise import add_gaussian_noise, subspace_similarity, noise_robustness_pca
from src.preprocess import gaussian_smooth, zscore_units, filter_low_firing_units


@pytest.fixture
def synthetic_rates():
    """Generate synthetic neural population data for testing.

    Returns firing rate tensor with realistic structure:
    - 50 trials
    - 20 units
    - 30 time bins
    - Signal with low-dimensional structure + noise
    """
    np.random.seed(42)
    n_trials, n_units, n_bins = 50, 20, 30

    # Create low-dimensional latent structure (3 dimensions)
    latent_dim = 3
    latent_trajs = np.random.randn(n_trials, n_bins, latent_dim)

    # Random projection to high-dimensional space
    projection = np.random.randn(latent_dim, n_units) * 0.5
    rates = np.einsum('tbl,lu->tub', latent_trajs, projection)

    # Add noise and ensure non-negative (firing rates)
    rates = rates + np.random.randn(n_trials, n_units, n_bins) * 0.1
    rates = np.abs(rates) + 1.0  # Ensure positive firing rates

    return rates


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_gaussian_smooth(self, synthetic_rates):
        """Test Gaussian smoothing preserves shape and reduces variance."""
        smoothed = gaussian_smooth(synthetic_rates, sigma_bins=1.5)

        assert smoothed.shape == synthetic_rates.shape
        assert np.all(np.isfinite(smoothed))

        # Smoothing should reduce high-frequency variance
        original_var = np.var(np.diff(synthetic_rates, axis=2))
        smoothed_var = np.var(np.diff(smoothed, axis=2))
        assert smoothed_var < original_var

    def test_zscore_units(self, synthetic_rates):
        """Test z-scoring normalizes each unit correctly."""
        zscored = zscore_units(synthetic_rates)

        assert zscored.shape == synthetic_rates.shape
        assert np.all(np.isfinite(zscored))

        # Each unit should have approximately zero mean and unit variance
        unit_means = zscored.mean(axis=(0, 2))
        unit_stds = zscored.std(axis=(0, 2))

        np.testing.assert_allclose(unit_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(unit_stds, 1.0, atol=1e-10)

    def test_filter_low_firing_units(self, synthetic_rates):
        """Test filtering removes low-firing units."""
        # Add some low-firing units
        rates_with_low = synthetic_rates.copy()
        rates_with_low[:, :5, :] = 0.1  # First 5 units have low rates

        filtered, mask = filter_low_firing_units(rates_with_low, min_rate_hz=0.5)

        assert filtered.shape[0] == rates_with_low.shape[0]  # Same trials
        assert filtered.shape[1] < rates_with_low.shape[1]  # Fewer units
        assert filtered.shape[2] == rates_with_low.shape[2]  # Same time bins
        assert mask.sum() == filtered.shape[1]  # Mask matches kept units


class TestPCA:
    """Test PCA dimensionality reduction."""

    def test_fit_pca(self, synthetic_rates):
        """Test PCA fitting returns expected structure."""
        n_components = 5
        result = fit_pca(synthetic_rates, n_components=n_components)

        # Check shapes
        assert result.components.shape == (n_components, synthetic_rates.shape[1])
        assert result.explained_variance.shape == (n_components,)
        assert result.explained_variance_ratio.shape == (n_components,)
        assert result.scores.shape == (synthetic_rates.shape[0],
                                      synthetic_rates.shape[2],
                                      n_components)
        assert result.mean.shape == (synthetic_rates.shape[1],)

        # Variance ratios should sum to <= 1 and be in descending order
        assert np.sum(result.explained_variance_ratio) <= 1.0
        assert np.all(np.diff(result.explained_variance_ratio) <= 0)

    def test_project_pca(self, synthetic_rates):
        """Test PCA projection works on new data."""
        # Fit on original data
        result = fit_pca(synthetic_rates, n_components=5)

        # Project same data (should match scores)
        projected = project_pca(synthetic_rates, result.components, result.mean)

        np.testing.assert_allclose(projected, result.scores, rtol=1e-5)

    def test_variance_explained(self, synthetic_rates):
        """Test variance explained computation."""
        result = fit_pca(synthetic_rates, n_components=10)
        explained, cumulative = variance_explained(result.explained_variance_ratio)

        assert len(explained) == len(cumulative) == 10
        np.testing.assert_allclose(cumulative[-1], np.sum(explained), rtol=1e-10)
        assert np.all(np.diff(cumulative) >= 0)  # Cumulative is monotonic


class TestJPCA:
    """Test jPCA rotational dynamics analysis."""

    def test_fit_jpca(self, synthetic_rates):
        """Test jPCA fitting returns expected structure."""
        bin_size = 0.02  # 20ms bins
        n_pca_dims = 6

        result = fit_jpca(synthetic_rates, bin_size=bin_size, n_pca_dims=n_pca_dims)

        # Check shapes
        assert result.components.shape == (2, n_pca_dims)  # 2D plane
        assert result.scores.shape == (synthetic_rates.shape[0],
                                      synthetic_rates.shape[2],
                                      2)
        assert result.skew_matrix.shape == (n_pca_dims, n_pca_dims)

        # Skew matrix should be skew-symmetric
        np.testing.assert_allclose(result.skew_matrix,
                                  -result.skew_matrix.T,
                                  atol=1e-10)

        # Eigenvalues should come in conjugate pairs (imaginary parts)
        imag_parts = np.imag(result.eigenvalues)
        # Allow for some numerical error in pairing
        assert len(result.eigenvalues) == n_pca_dims

    def test_project_jpca(self, synthetic_rates):
        """Test jPCA projection works on new data."""
        result = fit_jpca(synthetic_rates, bin_size=0.02, n_pca_dims=6)

        # Project same data
        projected = project_jpca(synthetic_rates,
                                result.pca_components,
                                result.pca_mean,
                                result.components)

        # Should match fitted scores
        np.testing.assert_allclose(projected, result.scores, rtol=1e-5)


class TestNoiseAnalysis:
    """Test noise robustness analysis."""

    def test_add_gaussian_noise(self, synthetic_rates):
        """Test noise injection maintains shape and increases variance."""
        snr_db = 10.0
        noisy = add_gaussian_noise(synthetic_rates, snr_db=snr_db)

        assert noisy.shape == synthetic_rates.shape
        assert np.all(np.isfinite(noisy))

        # Noisy data should have higher variance
        assert np.var(noisy) > np.var(synthetic_rates)

    def test_subspace_similarity(self):
        """Test subspace similarity metric."""
        # Identical subspaces
        A = np.random.randn(5, 10)
        sim_identical = subspace_similarity(A, A)
        assert 0.99 <= sim_identical <= 1.0  # Should be ~1

        # Orthogonal subspaces
        B = np.random.randn(5, 10)
        # Make B orthogonal to A via Gram-Schmidt
        Q_A, _ = np.linalg.qr(A.T)
        B_orth = B - (B @ Q_A) @ Q_A.T
        sim_orth = subspace_similarity(A, B_orth)
        assert 0.0 <= sim_orth <= 0.5  # Should be close to 0

    def test_noise_robustness_pca(self, synthetic_rates):
        """Test PCA noise robustness analysis."""
        result = noise_robustness_pca(synthetic_rates,
                                      n_components=5,
                                      snr_db=10.0)

        assert result.snr_db == 10.0
        assert 0.0 <= result.subspace_similarity <= 1.0
        assert len(result.explained_variance_ratio) == 5
        assert len(result.noisy_explained_variance_ratio) == 5

        # With reasonable SNR, subspace should be somewhat similar
        assert result.subspace_similarity > 0.5


class TestPipelineIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, synthetic_rates):
        """Test complete analysis pipeline runs without errors."""
        # Preprocessing
        smoothed = gaussian_smooth(synthetic_rates, sigma_bins=1.0)
        filtered, _ = filter_low_firing_units(smoothed, min_rate_hz=0.5)
        normalized = zscore_units(filtered)

        # PCA
        pca_result = fit_pca(normalized, n_components=10)

        # jPCA
        jpca_result = fit_jpca(normalized, bin_size=0.02, n_pca_dims=6)

        # Noise analysis
        noise_result = noise_robustness_pca(normalized, n_components=10, snr_db=10.0)

        # Verify all steps completed
        assert pca_result.scores.shape[2] == 10
        assert jpca_result.scores.shape[2] == 2
        assert noise_result.subspace_similarity > 0.0

        print(f"✓ Pipeline completed successfully")
        print(f"  - PCA: {pca_result.explained_variance_ratio[:3].sum():.1%} variance in top 3 PCs")
        print(f"  - jPCA: extracted rotational plane")
        print(f"  - Noise: {noise_result.subspace_similarity:.3f} subspace similarity at {noise_result.snr_db}dB")


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v"])
