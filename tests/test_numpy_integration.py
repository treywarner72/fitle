"""Tests for numpy integration with fitle Models and Params.

Tests the __array_ufunc__ and __array_function__ protocols that allow
numpy functions to work directly with Model and Param objects.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from fitle import INPUT
from fitle.param import Param
from fitle.model import Model


class TestNumpyUfuncs:
    """Test numpy ufuncs (np.exp, np.sin, etc.) with Models and Params."""

    def test_exp_with_input(self):
        """np.exp(INPUT) creates a Model."""
        model = np.exp(INPUT)
        assert isinstance(model, Model)
        assert_almost_equal(model(0), 1.0)
        assert_almost_equal(model(1), np.e)

    def test_exp_with_param(self):
        """np.exp(Param) creates a Model."""
        p = Param('x')(2.0)
        model = np.exp(p)
        assert isinstance(model, Model)
        assert_almost_equal(model(), np.exp(2.0))

    def test_sqrt_with_input(self):
        """np.sqrt(INPUT) creates a Model."""
        model = np.sqrt(INPUT)
        assert isinstance(model, Model)
        assert_almost_equal(model(4), 2.0)
        assert_array_almost_equal(model([1, 4, 9]), [1, 2, 3])

    def test_sin_cos_with_input(self):
        """Trig functions work with INPUT."""
        sin_model = np.sin(INPUT)
        cos_model = np.cos(INPUT)
        assert_almost_equal(sin_model(0), 0.0)
        assert_almost_equal(cos_model(0), 1.0)
        assert_almost_equal(sin_model(np.pi/2), 1.0)

    def test_ufunc_with_array_input(self):
        """Ufuncs work with array inputs."""
        model = np.exp(INPUT)
        x = np.array([0, 1, 2])
        result = model(x)
        assert_array_almost_equal(result, np.exp(x))

    def test_ufunc_scalar_input(self):
        """Ufuncs work with scalar inputs (converted to 0-d array)."""
        model = np.exp(INPUT)
        assert_almost_equal(model(0), 1.0)
        assert_almost_equal(model(1), np.e)

    def test_chained_ufuncs(self):
        """Multiple ufuncs can be chained."""
        model = np.exp(np.sin(INPUT))
        x = np.array([0, np.pi/2, np.pi])
        expected = np.exp(np.sin(x))
        assert_array_almost_equal(model(x), expected)


class TestNumpyArrayFunctions:
    """Test numpy array functions (np.sum, np.mean, etc.) with Models."""

    def test_mean_with_input(self):
        """np.mean(INPUT) creates a Model."""
        model = np.mean(INPUT)
        assert isinstance(model, Model)
        assert_almost_equal(model([1, 2, 3]), 2.0)

    def test_sum_with_input(self):
        """np.sum(INPUT) creates a Model."""
        model = np.sum(INPUT)
        assert isinstance(model, Model)
        assert_almost_equal(model([1, 2, 3]), 6.0)

    def test_mean_scalar_input(self):
        """np.mean works with scalar input."""
        model = np.mean(INPUT)
        assert_almost_equal(model(5), 5.0)
        assert_almost_equal(model(1), 1.0)

    def test_sum_scalar_input(self):
        """np.sum works with scalar input."""
        model = np.sum(INPUT)
        assert_almost_equal(model(5), 5.0)

    def test_dot_with_input(self):
        """np.dot works with INPUT."""
        model = np.dot(INPUT, INPUT)
        assert isinstance(model, Model)
        assert_almost_equal(model([1, 2, 3]), 14.0)  # 1+4+9

    def test_where_with_input(self):
        """np.where works with INPUT and creates Model."""
        model = np.where(INPUT > 0, INPUT, 0)
        assert isinstance(model, Model)
        result = model([-1, 0, 1, 2])
        assert_array_almost_equal(result, [0, 0, 1, 2])

    def test_where_with_param(self):
        """np.where works with Param threshold."""
        threshold = Param('t')(0)
        model = np.where(INPUT > threshold, INPUT, 0)
        assert isinstance(model, Model)
        result = model([-1, 0, 1, 2])
        assert_array_almost_equal(result, [0, 0, 1, 2])


class TestNumpyWithParams:
    """Test numpy functions with Param objects."""

    def test_param_in_expression(self):
        """Params can be used in numpy expressions."""
        a = Param('a')(2.0)
        b = Param('b')(3.0)
        model = np.exp(a * INPUT + b)
        assert isinstance(model, Model)
        assert_almost_equal(model(0), np.exp(3.0))
        assert_almost_equal(model(1), np.exp(5.0))

    def test_multiple_params(self):
        """Multiple params work together."""
        mu = Param('mu')(0)
        sigma = Param('sigma')(1)
        # Gaussian-like expression
        model = np.exp(-0.5 * ((INPUT - mu) / sigma) ** 2)
        assert_almost_equal(model(0), 1.0)
        assert_almost_equal(model(1), np.exp(-0.5))

    def test_param_value_changes(self):
        """Changing param value affects evaluation."""
        a = Param('a')(1.0)
        model = a * INPUT
        assert_almost_equal(model(5), 5.0)
        a.value = 2.0
        assert_almost_equal(model(5), 10.0)


class TestNumpyCompilation:
    """Test compilation of numpy-based Models."""

    def setup_method(self):
        """Clear cache before each test."""
        Model.clear_cache()

    def test_compile_exp(self):
        """np.exp model compiles successfully."""
        model = np.exp(INPUT)
        model.compile()
        assert model.compiled
        assert_almost_equal(model(0), 1.0)

    def test_compile_mean(self):
        """np.mean model compiles successfully."""
        model = np.mean(INPUT)
        model.compile()
        assert model.compiled
        assert_almost_equal(model([1, 2, 3, 4, 5]), 3.0)

    def test_compile_sum(self):
        """np.sum model compiles successfully."""
        model = np.sum(INPUT)
        model.compile()
        assert model.compiled
        assert_almost_equal(model([1, 2, 3]), 6.0)

    def test_compile_chained(self):
        """Chained numpy operations compile."""
        model = np.exp(-INPUT ** 2 / 2)
        model.compile()
        assert model.compiled
        assert_almost_equal(model(0), 1.0)

    def test_compile_with_params(self):
        """Models with params compile."""
        a = Param('a')(2.0)
        model = np.exp(a * INPUT)
        model.compile()
        assert model.compiled
        assert_almost_equal(model(0), 1.0)
        assert_almost_equal(model(1), np.exp(2.0))

    def test_compile_complex_expression(self):
        """Complex expressions compile."""
        mu = Param('mu')(0)
        sigma = Param.positive('sigma')(1)
        model = np.exp(-0.5 * ((INPUT - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        model.compile()
        assert model.compiled
        # Check it's a proper Gaussian at x=0
        assert_almost_equal(model(0), 1 / np.sqrt(2 * np.pi))

    def test_compiled_scalar_input(self):
        """Compiled model works with scalar input."""
        model = np.mean(INPUT)
        model.compile()
        assert_almost_equal(model(1), 1.0)
        assert_almost_equal(model(5), 5.0)

    def test_compiled_matches_python(self):
        """Compiled result matches Python evaluation."""
        model = np.exp(-INPUT ** 2)
        x = np.linspace(-2, 2, 10)

        # Python eval
        python_result = model.eval(x, {}, numba=False)

        # Compiled
        model.compile()
        compiled_result = model(x)

        assert_array_almost_equal(python_result, compiled_result)


class TestNumpyKwargsError:
    """Test that kwargs in numpy functions raise clear errors at compile time.

    Note: kwargs containing Model/Param are NOT fully supported. The values
    inside kwargs are captured in a closure but not tracked or evaluated
    as part of the Model tree. This means:
    - Python eval may work for simple cases (scalar kwargs)
    - Python eval will fail/recurse for Model/Param kwargs
    - Compilation always raises a clear error
    """

    def setup_method(self):
        """Clear cache before each test."""
        Model.clear_cache()

    def test_mean_with_where_compile_error(self):
        """np.mean with where= raises TypeError on compile."""
        model = np.mean(INPUT, where=INPUT > 0)
        with pytest.raises(TypeError) as exc_info:
            model.compile()
        assert "keyword arguments" in str(exc_info.value)
        assert "where" in str(exc_info.value)

    def test_sum_with_where_compile_error(self):
        """np.sum with where= raises TypeError on compile."""
        model = np.sum(INPUT, where=INPUT > 0)
        with pytest.raises(TypeError) as exc_info:
            model.compile()
        assert "keyword arguments" in str(exc_info.value)

    def test_kwargs_with_param_compile_error(self):
        """Kwargs containing Param also raise error on compile."""
        threshold = Param('t')(0)
        model = np.mean(INPUT, where=INPUT > threshold)
        with pytest.raises(TypeError) as exc_info:
            model.compile()
        assert "keyword arguments" in str(exc_info.value)

    def test_nested_kwargs_compile_error(self):
        """Kwargs in nested expression raise error."""
        inner = np.mean(INPUT, where=INPUT > 0)
        outer = np.exp(inner)
        with pytest.raises(TypeError) as exc_info:
            outer.compile()
        assert "keyword arguments" in str(exc_info.value)


class TestNumpyEdgeCases:
    """Edge cases and potential gotchas."""

    def setup_method(self):
        """Clear cache before each test."""
        Model.clear_cache()

    def test_empty_array(self):
        """Models handle empty arrays."""
        model = np.sum(INPUT)
        result = model(np.array([]))
        assert result == 0.0

    def test_single_element(self):
        """Models handle single-element arrays."""
        model = np.mean(INPUT)
        assert_almost_equal(model([42]), 42.0)

    def test_2d_array(self):
        """Models work with 2D arrays."""
        model = np.sum(INPUT)
        arr = np.array([[1, 2], [3, 4]])
        assert_almost_equal(model(arr), 10.0)

    def test_negative_values(self):
        """np.sqrt with negative values (should warn/nan)."""
        model = np.sqrt(INPUT)
        with np.errstate(invalid='ignore'):
            result = model([-1, 0, 1])
        assert np.isnan(result[0])
        assert_almost_equal(result[1], 0.0)
        assert_almost_equal(result[2], 1.0)

    def test_log_zero(self):
        """np.log with zero (should warn/-inf)."""
        model = np.log(INPUT)
        with np.errstate(divide='ignore'):
            result = model([0, 1, np.e])
        assert result[0] == -np.inf
        assert_almost_equal(result[1], 0.0)
        assert_almost_equal(result[2], 1.0)

    def test_model_reuse(self):
        """Same model can be evaluated multiple times."""
        model = np.exp(INPUT)
        r1 = model([0, 1])
        r2 = model([0, 1, 2])
        r3 = model(0)
        assert len(r1) == 2
        assert len(r2) == 3
        assert np.isscalar(r3) or r3.shape == ()

    def test_model_in_arithmetic(self):
        """Numpy models work in arithmetic expressions."""
        exp_model = np.exp(INPUT)
        combined = 2 * exp_model + 1
        assert isinstance(combined, Model)
        assert_almost_equal(combined(0), 3.0)  # 2*1 + 1

    def test_comparison_creates_model(self):
        """Comparisons with INPUT create Models."""
        model = INPUT > 0
        assert isinstance(model, Model)
        result = model([-1, 0, 1])
        assert_array_almost_equal(result, [False, False, True])

    def test_comparison_with_param(self):
        """Comparisons with Param create Models."""
        threshold = Param('t')(0.5)
        model = INPUT > threshold
        result = model([0, 0.5, 1])
        assert_array_almost_equal(result, [False, False, True])
