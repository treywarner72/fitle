# tests/test_pdfs.py
"""Tests for probability density functions."""
import pytest
import numpy as np
from fitle import Param, INPUT, gaussian, exponential, crystalball, convolve, index, const


class TestGaussian:
    """Gaussian/Normal distribution."""

    def test_gaussian_default_params(self):
        # Default: mu and sigma are Params
        model = gaussian()
        assert len(model.params) == 2

    def test_gaussian_with_params(self):
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)
        assert mu in model.params
        assert sigma in model.params

    def test_gaussian_at_mean(self):
        # Gaussian peaks at mu with value 1/(sigma*sqrt(2*pi))
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        expected_peak = 1 / (1 * np.sqrt(2 * np.pi))
        assert abs(model(0) - expected_peak) < 1e-10

    def test_gaussian_symmetric(self):
        mu = Param('mu')(5)
        sigma = Param('sigma')(2)
        model = gaussian(mu, sigma)

        # f(mu - x) == f(mu + x)
        assert abs(model(3) - model(7)) < 1e-10
        assert abs(model(1) - model(9)) < 1e-10

    def test_gaussian_normalization(self):
        # Integral should be ~1
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        integral = np.sum(model(x)) * dx

        assert abs(integral - 1.0) < 0.01

    def test_gaussian_different_sigma(self):
        mu = Param('mu')(0)
        sigma = Param('sigma')(2)
        model = gaussian(mu, sigma)

        # Peak should be 1/(2*sqrt(2*pi))
        expected_peak = 1 / (2 * np.sqrt(2 * np.pi))
        assert abs(model(0) - expected_peak) < 1e-10

    def test_gaussian_array_input(self):
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        x = np.array([-1, 0, 1])
        result = model(x)
        assert result.shape == (3,)
        assert result[1] > result[0]  # Peak at 0
        assert result[1] > result[2]


class TestExponential:
    """Exponential distribution."""

    def test_exponential_default_param(self):
        model = exponential()
        assert len(model.params) == 1

    def test_exponential_with_param(self):
        tau = Param('tau')(1)
        model = exponential(tau)
        assert tau in model.params

    def test_exponential_at_zero(self):
        tau = Param('tau')(2)
        model = exponential(tau)

        # f(0) = 1/tau
        expected = 1 / 2
        assert abs(model(0) - expected) < 1e-10

    def test_exponential_decay(self):
        tau = Param('tau')(1)
        model = exponential(tau)

        # f(x) should decrease with x
        x = np.array([0, 1, 2, 3])
        result = model(x)
        assert all(result[i] > result[i+1] for i in range(len(result)-1))

    def test_exponential_normalization(self):
        # Integral from 0 to inf should be 1
        tau = Param('tau')(1)
        model = exponential(tau)

        x = np.linspace(0, 20, 10000)
        dx = x[1] - x[0]
        integral = np.sum(model(x)) * dx

        assert abs(integral - 1.0) < 0.01

    def test_exponential_mean(self):
        # Mean of exponential is tau
        tau = Param('tau')(2)
        model = exponential(tau)

        x = np.linspace(0, 30, 10000)
        dx = x[1] - x[0]
        mean = np.sum(x * model(x)) * dx

        assert abs(mean - 2.0) < 0.1


class TestCrystalBall:
    """Crystal Ball distribution."""

    def test_crystalball_params(self):
        alpha = Param('alpha')(1)
        n = Param('n')(2)
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = crystalball(alpha, n, mu, sigma)

        assert alpha in model.params
        assert n in model.params
        assert mu in model.params
        assert sigma in model.params

    def test_crystalball_gaussian_core(self):
        # For x > mu - alpha*sigma, should behave like Gaussian
        alpha = Param('alpha')(2)
        n = Param('n')(3)
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)

        cb = crystalball(alpha, n, mu, sigma)
        gauss = gaussian(mu, sigma)

        # At x = 0 (well within Gaussian region), shapes should match
        # (up to normalization)
        x = np.array([0.0])
        cb_val = cb(x)
        gauss_val = gauss(x)

        # Both should be positive and at their peaks
        assert cb_val > 0
        assert gauss_val > 0

    def test_crystalball_has_tail(self):
        # For x < mu - alpha*sigma, should have power-law tail
        alpha = Param('alpha')(1)
        n = Param('n')(2)
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)

        model = crystalball(alpha, n, mu, sigma)

        # Deep in the tail
        x = np.array([-5.0, -4.0, -3.0])
        result = model(x)

        # Should still be positive
        assert all(result > 0)

        # Should decrease as we go further into tail
        assert result[0] < result[1] < result[2]

    def test_crystalball_continuous(self):
        # Check continuity at transition point
        alpha = Param('alpha')(1.5)
        n = Param('n')(3)
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)

        model = crystalball(alpha, n, mu, sigma)

        # Transition at t = -alpha, so x = mu - alpha*sigma = -1.5
        x_transition = -1.5
        eps = 0.001

        left = model(x_transition - eps)
        right = model(x_transition + eps)

        # Should be approximately continuous
        assert abs(left - right) / max(left, right) < 0.01

    def test_crystalball_normalization(self):
        # Integral should be ~1
        alpha = Param('alpha')(1.5)
        n = Param('n')(5)
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)

        model = crystalball(alpha, n, mu, sigma)

        x = np.linspace(-10, 5, 10000)
        dx = x[1] - x[0]
        integral = np.sum(model(x)) * dx

        assert abs(integral - 1.0) < 0.05


class TestConvolve:
    """Discrete convolution of histogram with Gaussian."""

    def test_convolve_basic(self):
        # Simple histogram
        centers = np.array([0.0, 1.0, 2.0])
        counts = np.array([1.0, 2.0, 1.0])

        mass_mother = 1.0  # Fixed value
        mu = Param('mu')(1)
        sigma = Param('sigma')(0.5)

        model = convolve(centers, counts, mass_mother, mu, sigma)

        # Should be evaluable
        x = np.linspace(-1, 4, 100)
        result = model(x)

        assert len(result) == len(x)
        assert all(np.isfinite(result))

    def test_convolve_peaked(self):
        # Delta-like histogram -> result should look like Gaussian
        centers = np.array([0.0])
        counts = np.array([1.0])

        mass_mother = 0.0
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)

        model = convolve(centers, counts, mass_mother, mu, sigma)

        x = np.linspace(-5, 5, 101)
        result = model(x)

        # Should peak near 0
        peak_idx = np.argmax(result)
        assert abs(x[peak_idx]) < 0.5

    def test_convolve_preserves_area(self):
        # Convolution should preserve total probability (approximately)
        centers = np.array([0.0, 1.0, 2.0])
        counts = np.array([100.0, 200.0, 100.0])

        mass_mother = 1.0
        mu = Param('mu')(1)
        sigma = Param('sigma')(0.3)

        x = np.linspace(-2, 5, 1000)
        dx = x[1] - x[0]

        # Pass bin_width matching evaluation grid for proper normalization
        model = convolve(centers, counts, mass_mother, mu, sigma, bin_width=dx)

        result = model(x)
        integral = np.sum(result) * dx

        # Normalized, so should integrate to ~1
        assert abs(integral - 1.0) < 0.1


class TestPDFEdgeCases:
    """Edge cases for PDFs."""

    def test_gaussian_very_small_sigma(self):
        mu = Param('mu')(0)
        sigma = Param('sigma')(0.01)
        model = gaussian(mu, sigma)

        # Should still be evaluable and peak at mu
        x = np.linspace(-0.1, 0.1, 1000)
        result = model(x)
        peak_idx = np.argmax(result)
        assert abs(x[peak_idx]) < 0.01

    def test_gaussian_large_sigma(self):
        mu = Param('mu')(0)
        sigma = Param('sigma')(100)
        model = gaussian(mu, sigma)

        # Should be very flat
        x = np.array([-10, 0, 10])
        result = model(x)
        # All values should be similar
        assert max(result) / min(result) < 1.1

    def test_exponential_large_tau(self):
        tau = Param('tau')(100)
        model = exponential(tau)

        x = np.array([0, 1, 2])
        result = model(x)
        # Should be nearly constant for small x
        assert max(result) / min(result) < 1.05

    def test_pdfs_with_array_params(self):
        # Params should work with their current values
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        result1 = model(0)

        mu.value = 1
        result2 = model(1)  # Now peak is at 1

        # Both should be at the peak
        assert abs(result1 - result2) < 1e-10
