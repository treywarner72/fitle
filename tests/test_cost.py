# tests/test_cost.py
"""Tests for cost functions."""
import pytest
import numpy as np
from fitle import Param, INPUT, Model, Cost, gaussian, const


class TestMSE:
    """Mean Squared Error cost function."""

    def test_mse_perfect_fit(self):
        # Model exactly matches data -> MSE = 0
        a = Param('a')(2)
        b = Param('b')(1)
        model = a * INPUT + b  # y = 2x + 1

        x = np.array([0, 1, 2, 3])
        y = np.array([1, 3, 5, 7])  # Exact values

        loss = model | Cost.MSE(x, y)
        assert loss() == 0

    def test_mse_nonzero(self):
        # Model doesn't match -> MSE > 0
        a = Param('a')(1)  # Wrong slope
        b = Param('b')(0)  # Wrong intercept
        model = a * INPUT + b  # y = x

        x = np.array([0, 1, 2])
        y = np.array([1, 3, 5])  # Actual: 2x + 1

        loss = model | Cost.MSE(x, y)
        # Predictions: [0, 1, 2], Actual: [1, 3, 5]
        # Errors: [1, 2, 3], Squared: [1, 4, 9], Sum: 14
        assert loss() == 14

    def test_mse_manual_calculation(self):
        a = Param('a')(0)
        model = a + INPUT  # y = x

        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])

        loss = model | Cost.MSE(x, y)
        # Predictions: [1, 2, 3], Actual: [2, 4, 6]
        # Errors squared: [1, 4, 9], Sum: 14
        assert loss() == 14


class TestNLL:
    """Unbinned Negative Log-Likelihood."""

    def test_nll_basic(self):
        # NLL = -sum(log(pdf(x)))
        # Constant model should broadcast to match data size
        model = const(0.5)

        x = np.array([0.5, 1.0, 1.5])
        loss = model | Cost.NLL(x)

        expected = -3 * np.log(0.5)
        assert abs(loss() - expected) < 1e-10

    def test_nll_gaussian(self):
        # Test with actual Gaussian
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        x = np.array([0.0])  # Single point at mean
        loss = model | Cost.NLL(x)

        # Gaussian at mean: 1/(sigma*sqrt(2*pi)) = 1/sqrt(2*pi)
        expected = -np.log(1 / np.sqrt(2 * np.pi))
        assert abs(loss() - expected) < 1e-6

    def test_nll_alias(self):
        # Cost.NLL is alias for Cost.unbinnedNLL
        # They reference the same underlying function
        assert Cost.NLL.__func__ is Cost.unbinnedNLL.__func__


class TestChi2:
    """Chi-squared cost function."""

    def test_chi2_from_raw_data(self):
        # Generate data and bin it
        # Use enough data and narrow range to avoid empty bins
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)

        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        loss = model | Cost.chi2(data=data, bins=10, range=(-2, 2))
        # Should be finite and positive
        assert np.isfinite(loss())
        assert loss() > 0

    def test_chi2_from_xy(self):
        # Provide pre-binned data
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        bin_widths = np.array([1.0, 1.0, 1.0, 1.0])

        # Model that predicts y/bin_width at each x
        a = Param('a')(10)
        model = a * INPUT  # Predicts 10, 20, 30, 40

        loss = model | Cost.chi2(x=x, y=y, bin_widths=bin_widths)
        # Perfect fit -> chi2 = 0
        assert abs(loss()) < 1e-10

    def test_chi2_nonzero(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 10.0, 10.0])
        bin_widths = np.array([1.0, 1.0, 1.0])

        a = Param('a')(5)  # Wrong value
        model = a + INPUT * 0  # Constant 5

        loss = model | Cost.chi2(x=x, y=y, bin_widths=bin_widths)
        # Predictions: 5*1=5 for each bin, Observed: 10 each
        # chi2 = sum((y - pred*width)^2 / y) = sum((10 - 5)^2 / 10) = 3 * 2.5 = 7.5
        assert abs(loss() - 7.5) < 1e-10

    def test_chi2_zero_method_error(self):
        # Default: raise error if any bin has zero counts
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 0.0, 10.0])  # Zero in middle
        bin_widths = np.array([1.0, 1.0, 1.0])

        model = const(1)
        with pytest.raises(ValueError, match="non-zero"):
            model | Cost.chi2(x=x, y=y, bin_widths=bin_widths)

    def test_chi2_zero_method_absolute(self):
        # zero_method='absolute' ignores zero bins
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 0.0, 10.0])
        bin_widths = np.array([1.0, 1.0, 1.0])

        a = Param('a')(10)
        model = a + INPUT * 0  # Constant 10

        loss = model | Cost.chi2(x=x, y=y, bin_widths=bin_widths, zero_method='absolute')
        # Should work without error
        assert np.isfinite(loss())

    def test_chi2_cannot_mix_data_and_xy(self):
        with pytest.raises(ValueError, match="Cannot provide both"):
            Cost.chi2(data=[1, 2, 3], bins=10, x=[1], y=[1])

    def test_chi2_requires_data_or_xy(self):
        with pytest.raises(ValueError):
            Cost.chi2()


class TestBinnedNLL:
    """Binned Negative Log-Likelihood."""

    def test_binned_nll_from_raw_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        model = gaussian(mu, sigma)

        loss = model | Cost.binnedNLL(data=data, bins=20, range=(-3, 3))
        assert np.isfinite(loss())
        assert loss() > 0

    def test_binned_nll_from_xy(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([100.0, 200.0, 100.0])
        bin_widths = np.array([1.0, 1.0, 1.0])

        a = Param('a')(100)
        model = a + INPUT * 0  # Constant

        loss = model | Cost.binnedNLL(x=x, y=y, bin_widths=bin_widths)
        assert np.isfinite(loss())

    def test_binned_nll_requires_bin_widths_for_xy(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="bin_widths"):
            Cost.binnedNLL(x=x, y=y)

    def test_binned_nll_cannot_mix_data_and_xy(self):
        with pytest.raises(ValueError, match="Cannot provide both"):
            Cost.binnedNLL(data=[1, 2, 3], bins=10, x=[1], y=[1])


class TestCostPipeOperator:
    """Test the | pipe operator for costs."""

    def test_pipe_returns_model(self):
        a = Param('a')(1)
        model = a * INPUT

        loss = model | Cost.MSE([1, 2], [2, 4])
        assert isinstance(loss, Model)

    def test_pipe_preserves_base_model(self):
        a = Param('a')(1)
        model = a * INPUT

        loss = model | Cost.MSE([1, 2], [2, 4])
        assert loss.memory['base'] is model

    def test_pipe_preserves_cost(self):
        a = Param('a')(1)
        model = a * INPUT

        cost = Cost.MSE([1, 2], [2, 4])
        loss = model | cost
        assert loss.memory['cost'] is cost

    def test_pipe_result_evaluates_to_scalar(self):
        a = Param('a')(1)
        model = a * INPUT

        loss = model | Cost.MSE([1, 2, 3], [1, 2, 3])
        result = loss()
        assert np.isscalar(result) or result.shape == ()


class TestBinWidthHandling:
    """Test bin width calculation and handling."""

    def test_chi2_infers_widths_from_centers(self):
        # Uniform spacing
        x = np.array([0.5, 1.5, 2.5, 3.5])
        y = np.array([10.0, 10.0, 10.0, 10.0])

        model = const(10)
        # Should infer widths as [1, 1, 1, 1]
        loss = model | Cost.chi2(x=x, y=y)
        assert np.isfinite(loss())

    def test_chi2_with_explicit_widths(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 10.0, 10.0])
        widths = np.array([0.5, 1.0, 1.5])

        model = const(10)
        loss = model | Cost.chi2(x=x, y=y, bin_widths=widths)
        assert np.isfinite(loss())

    def test_chi2_single_bin_requires_explicit_width(self):
        x = np.array([1.0])
        y = np.array([10.0])

        model = const(10)
        with pytest.raises(ValueError, match="single bin"):
            model | Cost.chi2(x=x, y=y)


class TestCostWithParams:
    """Test that cost functions work with model parameters."""

    def test_cost_has_model_params(self):
        a = Param('a')(1)
        b = Param('b')(2)
        model = a * INPUT + b

        loss = model | Cost.MSE([1, 2], [3, 5])
        assert a in loss.params
        assert b in loss.params

    def test_cost_value_changes_with_params(self):
        a = Param('a')(1)
        model = a * INPUT

        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])  # Matches a=2

        loss = model | Cost.MSE(x, y)

        # With a=1: predictions [1,2,3], errors [1,2,3], MSE=14
        assert loss() == 14

        # Change param
        a.value = 2
        # With a=2: predictions [2,4,6], errors [0,0,0], MSE=0
        assert loss() == 0
