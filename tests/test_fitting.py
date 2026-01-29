# tests/test_fitting.py
"""End-to-end fitting tests."""
import pytest
import numpy as np
from fitle import Param, INPUT, fit, FitResult, Cost, gaussian, exponential, const


class TestParameterRecovery:
    """Test that fitting recovers known parameters."""

    def test_linear_mse(self):
        # y = 2x + 3
        np.random.seed(42)
        true_a, true_b = 2.0, 3.0

        x = np.linspace(0, 10, 100)
        y = true_a * x + true_b + np.random.normal(0, 0.5, len(x))

        a = Param('a')(0, 5)(1)
        b = Param('b')(0, 10)(1)
        model = a * INPUT + b

        result = fit(model | Cost.MSE(x, y))

        assert result.success
        assert abs(result.values['a'] - true_a) < 0.2
        assert abs(result.values['b'] - true_b) < 0.5

    def test_gaussian_nll(self):
        # Recover mu and sigma from Gaussian data
        np.random.seed(42)
        true_mu, true_sigma = 5.0, 1.5

        data = np.random.normal(true_mu, true_sigma, 10000)

        mu = Param('mu')(0, 10)(4)
        sigma = Param.positive('sigma')(1)
        model = gaussian(mu, sigma)

        result = fit(model | Cost.NLL(data))

        assert result.success
        assert abs(result.values['mu'] - true_mu) < 0.1
        assert abs(result.values['sigma'] - true_sigma) < 0.1

    def test_gaussian_chi2(self):
        # Recover parameters using chi2 with binned data
        # Must include normalization N since chi2 compares counts
        np.random.seed(42)
        true_mu, true_sigma = 0.0, 1.0
        n_events = 50000

        data = np.random.normal(true_mu, true_sigma, n_events)

        mu = Param('mu')(-2, 2)(0.5)
        sigma = Param.positive('sigma')(0.5)
        N = Param.positive('N')(n_events)
        model = N * gaussian(mu, sigma)

        result = fit(model | Cost.chi2(data=data, bins=50, range=(-4, 4)))

        assert result.success
        assert abs(result.values['mu'] - true_mu) < 0.1
        assert abs(result.values['sigma'] - true_sigma) < 0.1
        assert abs(result.values['N'] - n_events) / n_events < 0.1

    def test_gaussian_binned_nll(self):
        np.random.seed(42)
        true_mu, true_sigma = 2.0, 0.8

        data = np.random.normal(true_mu, true_sigma, 20000)

        mu = Param('mu')(0, 5)(1)
        sigma = Param.positive('sigma')(1)
        model = gaussian(mu, sigma)

        result = fit(model | Cost.binnedNLL(data=data, bins=40, range=(-1, 5)))

        assert result.success
        assert abs(result.values['mu'] - true_mu) < 0.1
        assert abs(result.values['sigma'] - true_sigma) < 0.1

    def test_exponential_nll(self):
        np.random.seed(42)
        true_tau = 2.5

        data = np.random.exponential(true_tau, 10000)

        tau = Param.positive('tau')(1)
        model = exponential(tau)

        result = fit(model | Cost.NLL(data))

        assert result.success
        assert abs(result.values['tau'] - true_tau) < 0.2

    def test_two_gaussians(self):
        # Mixture of two Gaussians
        np.random.seed(42)
        true_mu1, true_mu2 = -2.0, 2.0
        true_sigma = 0.5
        true_frac = 0.3

        n = 10000
        n1 = int(n * true_frac)
        data = np.concatenate([
            np.random.normal(true_mu1, true_sigma, n1),
            np.random.normal(true_mu2, true_sigma, n - n1)
        ])

        mu1 = Param('mu1')(-5, 0)(-1)
        mu2 = Param('mu2')(0, 5)(1)
        sigma = Param.positive('sigma')(1)
        frac = Param.unit('frac')(0.5)

        model = frac * gaussian(mu1, sigma) + (1 - frac) * gaussian(mu2, sigma)

        result = fit(model | Cost.NLL(data))

        assert result.success
        # Check parameters recovered (order might swap)
        mus = sorted([result.values['mu1'], result.values['mu2']])
        assert abs(mus[0] - true_mu1) < 0.3
        assert abs(mus[1] - true_mu2) < 0.3


class TestFitResult:
    """Test FitResult object attributes."""

    def test_fval_finite(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        model = gaussian(Param('mu')(0), Param.positive('sigma')(1))

        result = fit(model | Cost.NLL(data))

        assert np.isfinite(result.fval)

    def test_success_flag(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        model = gaussian(Param('mu')(0), Param.positive('sigma')(1))

        result = fit(model | Cost.NLL(data))

        assert isinstance(result.success, bool)
        assert result.success

    def test_values_dict(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)
        model = gaussian(mu, sigma)

        result = fit(model | Cost.NLL(data))

        assert 'mu' in result.values
        assert 'sigma' in result.values
        assert isinstance(result.values['mu'], float)
        assert isinstance(result.values['sigma'], float)

    def test_errors_dict(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)
        model = gaussian(mu, sigma)

        result = fit(model | Cost.NLL(data))

        assert 'mu' in result.errors
        assert 'sigma' in result.errors
        assert result.errors['mu'] > 0
        assert result.errors['sigma'] > 0

    def test_predict(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 0.1, len(x))

        a = Param('a')(2)
        b = Param('b')(1)
        model = a * INPUT + b

        result = fit(model | Cost.MSE(x, y))

        # predict should work
        pred = result.predict(np.array([0, 5, 10]))
        assert len(pred) == 3
        assert all(np.isfinite(pred))

    def test_minimizer_accessible(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        model = gaussian(Param('mu')(0), Param.positive('sigma')(1))

        result = fit(model | Cost.NLL(data))

        # Should have minimizer attribute
        assert hasattr(result, 'minimizer')
        assert result.minimizer is not None

    def test_repr(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        model = gaussian(Param('mu')(0), Param.positive('sigma')(1))

        result = fit(model | Cost.NLL(data))

        r = repr(result)
        assert 'FitResult' in r
        assert 'fval' in r


class TestFitOptions:
    """Test fit() options."""

    def test_numba_true_vs_false(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mu1 = Param('mu')(0)
        sigma1 = Param.positive('sigma')(1)
        model1 = gaussian(mu1, sigma1)

        mu2 = Param('mu')(0)
        sigma2 = Param.positive('sigma')(1)
        model2 = gaussian(mu2, sigma2)

        result_numba = fit(model1 | Cost.NLL(data), numba=True)
        result_no_numba = fit(model2 | Cost.NLL(data), numba=False)

        # Should get similar results
        assert abs(result_numba.values['mu'] - result_no_numba.values['mu']) < 0.01
        assert abs(result_numba.values['sigma'] - result_no_numba.values['sigma']) < 0.01

    def test_grad_true_vs_false(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mu1 = Param('mu')(0)
        sigma1 = Param.positive('sigma')(1)
        model1 = gaussian(mu1, sigma1)

        mu2 = Param('mu')(0)
        sigma2 = Param.positive('sigma')(1)
        model2 = gaussian(mu2, sigma2)

        result_grad = fit(model1 | Cost.NLL(data), grad=True)
        result_no_grad = fit(model2 | Cost.NLL(data), grad=False)

        # Should get similar results
        assert abs(result_grad.values['mu'] - result_no_grad.values['mu']) < 0.1
        assert abs(result_grad.values['sigma'] - result_no_grad.values['sigma']) < 0.1


class TestFitEdgeCases:
    """Edge cases for fitting."""

    def test_param_at_bound(self):
        # True value at parameter bound
        np.random.seed(42)
        data = np.abs(np.random.normal(0, 1, 1000))  # Positive only

        # sigma must be positive, true value is 1
        sigma = Param.positive('sigma')(0.5)
        # Simple model: exponential-like
        model = (1 / sigma) * const(1)  # Constant, won't fit well but should run

        # This should at least not crash
        mu = Param('mu')(0)
        model = gaussian(mu, sigma)
        result = fit(model | Cost.NLL(data))

        assert np.isfinite(result.fval)

    def test_poor_starting_values(self):
        # Start far from true values
        np.random.seed(42)
        data = np.random.normal(10, 0.5, 5000)

        mu = Param('mu')(0, 20)(0)  # Start at 0, true is 10
        sigma = Param.positive('sigma')(5)  # Start at 5, true is 0.5

        result = fit(gaussian(mu, sigma) | Cost.NLL(data))

        # Should still converge
        assert result.success
        assert abs(result.values['mu'] - 10) < 0.5
        assert abs(result.values['sigma'] - 0.5) < 0.2

    def test_single_parameter(self):
        # Model with just one free parameter
        np.random.seed(42)
        true_mu = 3.0
        data = np.random.normal(true_mu, 1.0, 5000)

        mu = Param('mu')(0, 10)(1)
        sigma = Param.positive('sigma')(1)  # Fixed at 1
        sigma.min = sigma.max = 1.0  # Fix it

        model = gaussian(mu, sigma)
        result = fit(model | Cost.NLL(data))

        assert result.success
        assert abs(result.values['mu'] - true_mu) < 0.1

    def test_many_parameters(self):
        # Polynomial fit with many parameters
        np.random.seed(42)

        x = np.linspace(0, 1, 100)
        # y = 1 + 2x + 3x^2
        y = 1 + 2 * x + 3 * x**2 + np.random.normal(0, 0.1, len(x))

        a0 = Param('a0')(0, 5)(0.5)
        a1 = Param('a1')(0, 5)(0.5)
        a2 = Param('a2')(0, 5)(0.5)

        model = a0 + a1 * INPUT + a2 * INPUT**2

        result = fit(model | Cost.MSE(x, y))

        assert result.success
        assert abs(result.values['a0'] - 1) < 0.3
        assert abs(result.values['a1'] - 2) < 0.5
        assert abs(result.values['a2'] - 3) < 0.5


class TestFitWithDifferentCosts:
    """Test fitting works with all cost functions."""

    def test_fit_mse(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.normal(0, 1, len(x))

        a = Param('a')(1)
        model = a * INPUT

        result = fit(model | Cost.MSE(x, y))
        assert result.success
        assert abs(result.values['a'] - 2) < 0.5

    def test_fit_nll(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)

        result = fit(gaussian(mu, sigma) | Cost.NLL(data))
        assert result.success

    def test_fit_chi2(self):
        np.random.seed(42)
        n_events = 10000
        data = np.random.normal(0, 1, n_events)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)
        N = Param.positive('N')(n_events)

        result = fit(N * gaussian(mu, sigma) | Cost.chi2(data=data, bins=20, range=(-3, 3)))
        assert result.success

    def test_fit_binned_nll(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)

        result = fit(gaussian(mu, sigma) | Cost.binnedNLL(data=data, bins=20, range=(-3, 3)))
        assert result.success


class TestFitDof:
    """Test degrees of freedom calculation."""

    def test_dof_binned(self):
        np.random.seed(42)
        n_events = 10000
        data = np.random.normal(0, 1, n_events)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)
        N = Param.positive('N')(n_events)

        result = fit(N * gaussian(mu, sigma) | Cost.chi2(data=data, bins=20, range=(-3, 3)))

        # dof = n_bins - n_params = 20 - 3 = 17
        assert result.dof() == 17

    def test_reduced_chi2(self):
        np.random.seed(42)
        n_events = 50000
        data = np.random.normal(0, 1, n_events)

        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)
        N = Param.positive('N')(n_events)

        result = fit(N * gaussian(mu, sigma) | Cost.chi2(data=data, bins=50, range=(-3, 3)))

        # Reduced chi2 should be ~1 for good fit
        reduced_chi2 = result.fval / result.dof()
        assert 0.5 < reduced_chi2 < 2.0
